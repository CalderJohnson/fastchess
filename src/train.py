"""Training scheme for FastChessNet using games, puzzles, and self-play."""
import torch
from huggingface_hub import login
from datasets import load_dataset

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
    
    print(f"\nPositional dataset: {hp.TRAIN_POSITIONAL_POSITIONS:,} training positions, {hp.VAL_POSITIONAL_POSITIONS:,} validation positions")
    print(f"Puzzle dataset: {hp.TRAIN_PUZZLE_POSITIONS:,} training positions, {hp.VAL_PUZZLE_POSITIONS:,} validation positions")
    
    # Create fixed positional validation set (load into memory)
    val_positional_dataset = ChessValidationDataset(
        hf_dataset=val_positional_dataset_hf,
        num_positions=hp.VAL_POSITIONAL_POSITIONS,
        min_elo=hp.MIN_ELO
    )

    # Load puzzle dataset from HuggingFace
    puzzle_dataset_hf_train = load_dataset(
        "Lichess/chess-puzzles",
        split="train",
        streaming=True
    )
    
    # Training puzzles (skip validation positions)
    train_puzzle_dataset = ChessPuzzleDataset(
        hf_dataset=puzzle_dataset_hf_train,
        num_positions=hp.TRAIN_PUZZLE_POSITIONS,
        skip_positions=hp.VAL_PUZZLE_POSITIONS,
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
        num_positions=hp.VAL_PUZZLE_POSITIONS,
        skip_positions=0,
        min_rating=hp.MIN_PUZZLE_RATING
    )
    
    
    # Create mixed validation dataloader
    val_dataloader = MixedDataLoader(
        positional_dataset=val_positional_dataset,
        puzzle_dataset=val_puzzle_dataset,
        batch_size=hp.PT_BATCH_SIZE,
        tactical_ratio=hp.TACTICAL_RATIO
    )
    
    # Calculate validation batches
    val_batches = (hp.VAL_POSITIONAL_POSITIONS + hp.VAL_PUZZLE_POSITIONS) // hp.PT_BATCH_SIZE
    
    print(f"\nTraining set: {hp.TRAIN_POSITIONAL_POSITIONS:,} positional + {len(train_puzzle_dataset):,} tactical positions in {hp.PT_STEPS:,} batches per epoch")
    print(f"Validation set: {len(val_positional_dataset):,} positional + {len(val_puzzle_dataset):,} tactical positions in {val_batches:,} batches")
    print(f"Batch composition: {int((1-hp.TACTICAL_RATIO)*100)}% games, {int(hp.TACTICAL_RATIO*100)}% puzzles")
    
    # Initialize model
    model = FastChessNet().to(hp.DEVICE)

    # Configure pretraining optimizer and scheduler
    total_steps = hp.PT_EPOCHS * hp.PT_STEPS
    warmup_steps = int(0.05 * total_steps)

    optimizer_pt = torch.optim.AdamW(
        model.parameters(),
        lr=hp.PT_LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer_pt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer_pt,
                start_factor=0.01,
                total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_pt,
                T_max=total_steps - warmup_steps,
                eta_min=hp.PT_LR * 0.01
            )
        ],
        milestones=[warmup_steps]
    )

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
            max_positions=hp.TRAIN_POSITIONAL_POSITIONS,
            skip_positions=hp.VAL_POSITIONAL_POSITIONS
        )
        
        # Create mixed dataloader
        mixed_dataloader = MixedDataLoader(
            positional_dataset=train_dataset,
            puzzle_dataset=train_puzzle_dataset,
            batch_size=hp.PT_BATCH_SIZE,
            tactical_ratio=hp.TACTICAL_RATIO
        )
        
        # Train with mixed data
        total_loss, policy_loss, value_loss = pretrain_epoch(
            model, mixed_dataloader, optimizer_pt, scheduler, hp.DEVICE, hp.PT_STEPS
        )
        
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
