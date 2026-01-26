"""Pretraining pipeline for FastChessNet using historical chess games."""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import chess
from tqdm import tqdm
import hyperparams as hp
from model import FastChessNet
from util import ChessPositionEncoder, MoveEncoder
import gc


class ChessDataset(Dataset):
    """Dataset of chess positions from historical games."""
    
    def __init__(self, iterable_dataset=None, min_elo=1800, max_positions_per_game=None, chunk_size=1000):
        """
        Args:
            iterable_dataset: HuggingFace IterableDataset (only needed for first preprocessing)
            min_elo: Minimum average ELO to include game
            max_positions_per_game: Limit positions per game (None = all)
            chunk_size: Number of games to process at once
        """
        self.encoder = ChessPositionEncoder()
        self.move_encoder = MoveEncoder()
        self.chunk_dir = hp.PT_DATASET_PATH + "_chunks"
        
        # Check if preprocessed data exists
        if os.path.exists(self.chunk_dir) and len(os.listdir(self.chunk_dir)) > 0:
            print("Loading all chunks into memory...")
            self.positions = self._load_all_chunks()
            print(f"Loaded {len(self.positions)} positions into memory")
            return
        
        # Preprocess from scratch
        if iterable_dataset is None:
            raise ValueError("Must provide iterable_dataset for initial preprocessing")
        
        print("Preprocessing games in chunks...")
        os.makedirs(self.chunk_dir, exist_ok=True)
        
        chunk_idx = 0
        chunk_buffer = []
        game_count = 0
        skipped_games = 0
        
        for game in tqdm(iterable_dataset, desc="Preprocessing games"):
            # Filter by ELO
            try:
                avg_elo = (game['white_elo'] + game['black_elo']) / 2
                if avg_elo < min_elo:
                    skipped_games += 1
                    continue
            except: # Some are missing ELO info
                skipped_games += 1
                continue
            
            # Parse game
            board = chess.Board()
            moves_uci = game['moves_uci']
            
            # Determine outcome
            if game['winner'] == 'white':
                outcome = 1.0
            elif game['winner'] == 'black':
                outcome = -1.0
            else:
                outcome = 0.0
            
            # Extract positions
            game_positions = []
            for move_uci in moves_uci:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        # Store minimal data: FEN, move index, and value
                        move_idx = self.move_encoder.encode_move(move)
                        
                        # Flip outcome for black's perspective
                        value = outcome if board.turn == chess.WHITE else -outcome
                        
                        # Store compressed representation
                        game_positions.append({
                            'fen': board.fen(),  # Compact string representation
                            'move_idx': move_idx,
                            'value': value
                        })
                        
                        board.push(move)
                except:
                    break
            
            # Sample positions if requested
            if max_positions_per_game and len(game_positions) > max_positions_per_game:
                indices = np.random.choice(len(game_positions), max_positions_per_game, replace=False)
                game_positions = [game_positions[i] for i in sorted(indices)]
            
            chunk_buffer.extend(game_positions)
            game_count += 1
            
            # Save chunk when buffer is full
            if game_count % chunk_size == 0:
                chunk_path = os.path.join(self.chunk_dir, f'chunk_{chunk_idx}.pt')
                torch.save(chunk_buffer, chunk_path)
                print(f"\nSaved chunk {chunk_idx} with {len(chunk_buffer)} positions")
                chunk_buffer = []
                chunk_idx += 1
                gc.collect()  # Force garbage collection
            
            # Stop after N_GAMES
            if game_count >= hp.N_GAMES:
                break
        
        # Save remaining positions
        if chunk_buffer:
            chunk_path = os.path.join(self.chunk_dir, f'chunk_{chunk_idx}.pt')
            torch.save(chunk_buffer, chunk_path)
            print(f"\nSaved final chunk {chunk_idx} with {len(chunk_buffer)} positions")
        
        # Load all chunks into memory
        self.positions = self._load_all_chunks()
        print(f"\nTotal: {len(self.positions)} positions from {game_count} games with {skipped_games} skipped games.")
    
    def _load_all_chunks(self):
        """Load all chunks into memory at once."""
        chunk_files = sorted([f for f in os.listdir(self.chunk_dir) if f.endswith('.pt')])
        all_positions = []
        
        for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            chunk_data = torch.load(chunk_path)
            all_positions.extend(chunk_data)
        
        return all_positions
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        """Generate board state and legal mask on-the-fly from FEN."""
        pos = self.positions[idx]
        
        # Reconstruct board from FEN
        board = chess.Board(pos['fen'])
        
        # Encode board state
        state = self.encoder.encode_board(board)
        
        # Generate legal mask
        legal_mask = self.move_encoder.get_legal_mask(board)
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor([pos['move_idx']]),
            legal_mask,
            torch.FloatTensor([pos['value']])
        )


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for states, target_moves, legal_masks, target_values in pbar:
        states = states.to(device)
        target_moves = target_moves.to(device).squeeze(1)
        legal_masks = legal_masks.to(device)
        target_values = target_values.to(device).squeeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_pred = model(states, legal_masks)
        
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, target_moves)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred.squeeze(), target_values)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'policy': f'{policy_loss.item():.4f}',
            'value': f'{value_loss.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss / n, policy_loss_sum / n, value_loss_sum / n


def main():
    # Load or create dataset
    chunk_dir = hp.PT_DATASET_PATH + "_chunks"
    
    if os.path.exists(chunk_dir) and len(os.listdir(chunk_dir)) > 0:
        # Load from preprocessed chunks
        print("Using existing preprocessed data...")
        dataset = ChessDataset(
            min_elo=hp.MIN_ELO,
            max_positions_per_game=hp.MAX_POSITIONS_PER_GAME
        )
    else:
        # Preprocess from HuggingFace
        print("Loading dataset from HuggingFace...")
        iterable_ds = load_dataset(
            "angeluriot/chess_games",
            split="train",
            streaming=True
        )
        
        dataset = ChessDataset(
            iterable_dataset=iterable_ds,
            min_elo=hp.MIN_ELO,
            max_positions_per_game=hp.MAX_POSITIONS_PER_GAME,
            chunk_size=hp.CHUNK_SIZE
        )
    
    # Create train/valid split
    total_size = len(dataset)
    train_size = int(total_size * hp.PT_SPLIT)
    valid_size = total_size - train_size
    
    print(f"\nDataset split: {train_size} train, {valid_size} validation")
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(hp.SEED)  # Reproducible split
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=hp.PT_BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=hp.PT_BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # Initialize model
    model = FastChessNet().to(hp.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.PT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(hp.PT_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{hp.PT_EPOCHS}")
        
        total_loss, policy_loss, value_loss = train_epoch(
            model, train_dataloader, optimizer, hp.DEVICE
        )
        scheduler.step()
        
        print(f"Train - Loss: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        
        # Validation
        if (epoch + 1) % 2 == 0:
            model.eval()
            val_total_loss = 0
            val_policy_loss = 0
            val_value_loss = 0
            
            with torch.no_grad():
                for states, target_moves, legal_masks, target_values in valid_dataloader:
                    states = states.to(hp.DEVICE)
                    target_moves = target_moves.to(hp.DEVICE).squeeze(1)
                    legal_masks = legal_masks.to(hp.DEVICE)
                    target_values = target_values.to(hp.DEVICE).squeeze(1)
                    
                    policy_logits, value_pred = model(states, legal_masks)
                    
                    p_loss = F.cross_entropy(policy_logits, target_moves)
                    v_loss = F.mse_loss(value_pred.squeeze(), target_values)
                    loss = p_loss + v_loss
                    
                    val_total_loss += loss.item()
                    val_policy_loss += p_loss.item()
                    val_value_loss += v_loss.item()
            
            val_total_loss /= len(valid_dataloader)
            val_policy_loss /= len(valid_dataloader)
            val_value_loss /= len(valid_dataloader)
            
            print(f"Valid - Loss: {val_total_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), hp.PT_MODEL_PATH)
    print(f"\nTraining complete! Model saved at '{hp.PT_MODEL_PATH}'")


if __name__ == "__main__":
    main()
