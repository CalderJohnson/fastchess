"""Training scheme for FastChessNet using self-play."""
import torch
from huggingface_hub import login

import hyperparams as hp
from keys import HF_TOKEN
from model import FastChessNet
from torch.utils.data import DataLoader
from datasets import load_dataset
from pretrain import pretrain_epoch, validate, ChessStreamDataset, ChessValidationDataset
from selfplay import SelfPlayDataset, selfplay_train_iteration


if hp.USE_SEED:
    torch.manual_seed(hp.SEED)

login(HF_TOKEN)  # Log in to HuggingFace to access datasets


def main():
    # Load streaming dataset from HuggingFace
    print("Setting up streaming dataset from HuggingFace...")
    hf_dataset = load_dataset(
        "angeluriot/chess_games",
        split="train",
        streaming=True
    )
    
    # Calculate validation/training split based on PT_SPLIT
    # PT_SPLIT is the fraction for training, so (1 - PT_SPLIT) is for validation
    val_games = int(hp.N_GAMES * (1 - hp.PT_SPLIT))
    train_games = hp.N_GAMES - val_games
    
    print(f"\nDataset split: {train_games} training games, {val_games} validation games")
    
    # Create fixed validation set (load into memory)
    val_dataset = ChessValidationDataset(
        hf_dataset=hf_dataset,
        num_games=val_games,
        min_elo=hp.MIN_ELO,
        max_positions_per_game=hp.MAX_POSITIONS_PER_GAME
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hp.PT_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Estimate batches per epoch
    positions_per_game = hp.MAX_POSITIONS_PER_GAME if hp.MAX_POSITIONS_PER_GAME else 40
    estimated_positions = train_games * positions_per_game
    estimated_batches = estimated_positions // hp.PT_BATCH_SIZE
    
    print(f"Estimated ~{estimated_positions:,} training positions ({estimated_batches:,} batches) per epoch")
    print(f"Validation set: {len(val_dataset):,} positions")
    
    # Initialize model
    model = FastChessNet().to(hp.DEVICE)

    # Configure pretraining optimizer and scheduler
    optimizer_pt = torch.optim.AdamW(model.parameters(), lr=hp.PT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_pt, step_size=3, gamma=0.5)

    # Configure self play optimizer
    buffer = SelfPlayDataset(model)
    optimizer_sp = torch.optim.AdamW(model.parameters(), lr=hp.LR, weight_decay=1e-4)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(hp.PT_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{hp.PT_EPOCHS}")
        
        # Recreate streaming dataset for each epoch
        hf_dataset = load_dataset(
            "angeluriot/chess_games",
            split="train",
            streaming=True
        )
        train_dataset = ChessStreamDataset(
            hf_dataset=hf_dataset,
            min_elo=hp.MIN_ELO,
            max_positions_per_game=hp.MAX_POSITIONS_PER_GAME,
            max_games=train_games,  # Use calculated training games
            skip_games=val_games  # Skip validation games
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=hp.PT_BATCH_SIZE,
            num_workers=0
        )
        
        # Train
        total_loss, policy_loss, value_loss = pretrain_epoch(
            model, train_dataloader, optimizer_pt, hp.DEVICE, estimated_batches
        )
        scheduler.step()
        
        print(f"Train - Loss: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        
        # Validate every epoch
        val_loss, val_policy_loss, val_value_loss = validate(model, val_dataloader, hp.DEVICE)
        print(f"Valid - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")

        print("Selfplay training iteration...")
        for i in range(hp.ITERATIONS):
            print(f"  Iteration {i + 1}/{hp.ITERATIONS}")
            for _ in range(hp.GAMES_PER_ITER):
                buffer.self_play()
            if len(buffer) >= hp.BATCH_SIZE:
                selfplay_train_iteration(model, buffer, optimizer_sp)

        # Validate after self-play training to assess improvement/forgetting
        val_loss, val_policy_loss, val_value_loss = validate(model, val_dataloader, hp.DEVICE)
        print(f"Post-Selfplay Valid - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")

        if epoch % 2 == 0: # Save checkpoint every 2 epochs
            torch.save(model.state_dict(), hp.MODEL_PATH + f"checkpoint_{epoch//2}.pt")
    
    # Save final model
    torch.save(model.state_dict(), hp.PT_MODEL_PATH + "model_final.pt")
    print(f"\nTraining complete! Model saved at '{hp.PT_MODEL_PATH}'")

if __name__ == "__main__":
    main()
