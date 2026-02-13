"""Training scheme for FastChessNet using games, puzzles, and self-play."""
import torch
from huggingface_hub import login
from datasets import load_dataset
from torch.utils.data import DataLoader

import hyperparams as hp
from keys import HF_TOKEN
from model import FastChessNet
from pretrain import (
    pretrain_epoch, validate, 
    ChessStreamDataset, ChessValidationDataset, 
    ChessPuzzleDataset, MixedDataLoader
)
from selfplay import SelfPlayDataset, selfplay_train_iteration


if hp.USE_SEED:
    torch.manual_seed(hp.SEED)

login(HF_TOKEN)  # Log in to HuggingFace to access datasets


def main():
    # Load validation positional data dataset from HuggingFace (training data is streamed in pretrain_epoch)
    val_positional_dataset_hf = load_dataset(
        "angeluriot/chess_games",
        split="train",
        streaming=True
    )
    
    # Calculate validation/training split based on PT_SPLIT
    val_games = int(hp.N_GAMES * (1 - hp.PT_SPLIT))
    train_games = hp.N_GAMES - val_games
    
    print(f"\nPositional dataset split: {train_games} training games, {val_games} validation games")
    
    # Create fixed positional validation set (load into memory)
    val_positional_dataset = ChessValidationDataset(
        hf_dataset=val_positional_dataset_hf,
        num_games=val_games,
        min_elo=hp.MIN_ELO,
        max_positions_per_game=hp.MAX_POSITIONS_PER_GAME
    )

    # Load puzzle dataset from HuggingFace
    puzzle_dataset_hf_train = load_dataset(
        "Lichess/chess-puzzles",
        split="train",
        streaming=True
    )
    
    val_puzzles = int(hp.N_PUZZLES * (1 - hp.PT_SPLIT))
    train_puzzles = hp.N_PUZZLES - val_puzzles

    print(f"Puzzle split: {train_puzzles} training puzzles, {val_puzzles} validation puzzles")
    
    # Training puzzles
    train_puzzle_dataset = ChessPuzzleDataset(
        hf_dataset=puzzle_dataset_hf_train,
        num_puzzles=train_puzzles,
        skip_puzzles=val_puzzles,
        min_rating=hp.MIN_PUZZLE_RATING
    )
    
    # Validation puzzles
    puzzle_dataset_hf_val = load_dataset(
        "Lichess/chess-puzzles",
        split="train",
        streaming=True
    )
    
    val_puzzle_dataset = ChessPuzzleDataset(
        hf_dataset=puzzle_dataset_hf_val,
        num_puzzles=val_puzzles,
        skip_puzzles=0,
        min_rating=hp.MIN_PUZZLE_RATING
    )
    
    # Estimate batches per epoch
    estimated_positions = train_games * hp.MAX_POSITIONS_PER_GAME
    estimated_batches = estimated_positions * (1 / (1 - hp.TACTICAL_RATIO)) // hp.PT_BATCH_SIZE
    
    # Create mixed validation dataloader
    val_dataloader = MixedDataLoader(
        positional_dataset=val_positional_dataset,
        puzzle_dataset=val_puzzle_dataset,
        batch_size=hp.PT_BATCH_SIZE,
        tactical_ratio=hp.TACTICAL_RATIO
    )
    
    # Calculate validation batches
    val_positions = len(val_positional_dataset)
    val_batches = val_positions / (1 / (1 - hp.TACTICAL_RATIO)) // hp.PT_BATCH_SIZE
    
    print(f"\nTraining set: ~{estimated_positions:,} positional + {len(train_puzzle_dataset):,} tactical in {estimated_batches:,} batches per epoch")
    print(f"Validation set: {val_positions:,} positional + {len(val_puzzle_dataset):,} tactical in {val_batches:,} batches")
    print(f"Batch composition: {int((1-hp.TACTICAL_RATIO)*100)}% games, {int(hp.TACTICAL_RATIO*100)}% puzzles")
    
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
        positional_dataset = load_dataset(
            "angeluriot/chess_games",
            split="train",
            streaming=True
        )
        train_dataset = ChessStreamDataset(
            hf_dataset=positional_dataset,
            min_elo=hp.MIN_ELO,
            max_positions_per_game=hp.MAX_POSITIONS_PER_GAME,
            max_games=train_games,
            skip_games=val_games
        )
        
        # Create mixed dataloader (80% games, 20% puzzles)
        mixed_dataloader = MixedDataLoader(
            positional_dataset=train_dataset,
            puzzle_dataset=train_puzzle_dataset,
            batch_size=hp.PT_BATCH_SIZE,
            tactical_ratio=hp.TACTICAL_RATIO
        )
        
        # Train with mixed data
        total_loss, policy_loss, value_loss = pretrain_epoch(
            model, mixed_dataloader, optimizer_pt, hp.DEVICE, estimated_batches
        )
        scheduler.step()
        
        print(f"Train - Loss: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        
        # Validate every epoch
        val_loss, val_policy_loss, val_value_loss = validate(model, val_dataloader, hp.DEVICE, val_batches)
        print(f"Valid - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")

        # Self-play training iterations
        # print("Selfplay training iteration...")
        # for i in range(hp.ITERATIONS):
        #     print(f"  Iteration {i + 1}/{hp.ITERATIONS}")
        #     for _ in range(hp.GAMES_PER_ITER):
        #         buffer.self_play()
        #     if len(buffer) >= hp.BATCH_SIZE:
        #         selfplay_train_iteration(model, buffer, optimizer_sp)

        # # Validate after self-play training to assess improvement/forgetting
        # val_loss, val_policy_loss, val_value_loss = validate(model, val_dataloader, hp.DEVICE, val_batches)
        # print(f"Post-Selfplay Valid - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")

        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            checkpoint_path = hp.MODEL_PATH + f"checkpoint_{epoch//2}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = hp.PT_MODEL_PATH + "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved at '{final_path}'")


if __name__ == "__main__":
    main()
