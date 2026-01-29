
"""Pretraining pipeline for FastChessNet using historical chess games."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets import load_dataset
import chess
from tqdm import tqdm
import hyperparams as hp
from model import FastChessNet
from util import ChessPositionEncoder, MoveEncoder


class ChessValidationDataset(Dataset):
    """Fixed validation dataset loaded into memory."""
    
    def __init__(self, hf_dataset, num_games, min_elo=1800, max_positions_per_game=None):
        """
        Args:
            hf_dataset: HuggingFace IterableDataset
            num_games: Number of games to load for validation
            min_elo: Minimum average ELO to include game
            max_positions_per_game: Limit positions per game (None = all)
        """
        self.encoder = ChessPositionEncoder()
        self.move_encoder = MoveEncoder()
        self.positions = []
        
        print(f"Loading {num_games} games for validation set...")
        game_count = 0
        
        for game in tqdm(hf_dataset, desc="Loading validation games", total=num_games):
            if game_count >= num_games:
                break
            
            # Filter by ELO
            try:
                avg_elo = (game['white_elo'] + game['black_elo']) / 2
                if avg_elo < min_elo:
                    continue
            except:
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
                        move_idx = self.move_encoder.encode_move(move)
                        value = outcome if board.turn == chess.WHITE else -outcome
                        
                        # Store FEN for memory efficiency
                        game_positions.append({
                            'fen': board.fen(),
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
            
            self.positions.extend(game_positions)
            game_count += 1
        
        print(f"Loaded {len(self.positions)} validation positions from {game_count} games")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
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


class ChessStreamDataset(IterableDataset):
    """Streaming dataset that processes chess games on-the-fly from HuggingFace for training."""
    
    def __init__(self, hf_dataset, min_elo=1800, max_positions_per_game=None, 
                 max_games=None, skip_games=0):
        """
        Args:
            hf_dataset: HuggingFace IterableDataset
            min_elo: Minimum average ELO to include game
            max_positions_per_game: Limit positions per game (None = all)
            max_games: Maximum number of games to process (None = all)
            skip_games: Number of games to skip at start (for train/val split)
        """
        super().__init__()
        self.hf_dataset = hf_dataset
        self.min_elo = min_elo
        self.max_positions_per_game = max_positions_per_game
        self.max_games = max_games
        self.skip_games = skip_games
        self.encoder = ChessPositionEncoder()
        self.move_encoder = MoveEncoder()
    
    def __iter__(self):
        """Yield positions one at a time from the streaming dataset."""
        game_count = 0
        skipped_count = 0
        
        for game in self.hf_dataset:
            # Skip games for validation split
            if skipped_count < self.skip_games:
                # Still need to check ELO to skip the right games
                try:
                    avg_elo = (game['white_elo'] + game['black_elo']) / 2
                    if avg_elo >= self.min_elo:
                        skipped_count += 1
                except:
                    pass
                continue
            
            # Stop if max games reached
            if self.max_games and game_count >= self.max_games:
                break
            
            # Filter by ELO
            try:
                avg_elo = (game['white_elo'] + game['black_elo']) / 2
                if avg_elo < self.min_elo:
                    continue
            except:
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
                        move_idx = self.move_encoder.encode_move(move)
                        value = outcome if board.turn == chess.WHITE else -outcome
                        
                        # Encode and yield immediately
                        state = self.encoder.encode_board(board)
                        legal_mask = self.move_encoder.get_legal_mask(board)
                        
                        game_positions.append({
                            'state': state,
                            'move_idx': move_idx,
                            'legal_mask': legal_mask,
                            'value': value
                        })
                        
                        board.push(move)
                except:
                    break
            
            # Sample positions if requested
            if self.max_positions_per_game and len(game_positions) > self.max_positions_per_game:
                indices = np.random.choice(len(game_positions), self.max_positions_per_game, replace=False)
                game_positions = [game_positions[i] for i in sorted(indices)]
            
            # Yield all positions from this game
            for pos in game_positions:
                yield (
                    torch.FloatTensor(pos['state']),
                    torch.LongTensor([pos['move_idx']]),
                    pos['legal_mask'],
                    torch.FloatTensor([pos['value']])
                )
            
            game_count += 1


def train_epoch(model, dataloader, optimizer, device, total_batches=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    batch_count = 0
    
    pbar = tqdm(dataloader, desc="Training", total=total_batches)
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
        batch_count += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'policy': f'{policy_loss.item():.4f}',
            'value': f'{value_loss.item():.4f}'
        })
    
    return total_loss / batch_count, policy_loss_sum / batch_count, value_loss_sum / batch_count


def validate(model, dataloader, device):
    """Run validation."""
    model.eval()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    batch_count = 0
    
    with torch.no_grad():
        for states, target_moves, legal_masks, target_values in tqdm(dataloader, desc="Validating"):
            states = states.to(device)
            target_moves = target_moves.to(device).squeeze(1)
            legal_masks = legal_masks.to(device)
            target_values = target_values.to(device).squeeze(1)
            
            policy_logits, value_pred = model(states, legal_masks)
            
            p_loss = F.cross_entropy(policy_logits, target_moves)
            v_loss = F.mse_loss(value_pred.squeeze(), target_values)
            loss = p_loss + v_loss
            
            total_loss += loss.item()
            policy_loss_sum += p_loss.item()
            value_loss_sum += v_loss.item()
            batch_count += 1
    
    return total_loss / batch_count, policy_loss_sum / batch_count, value_loss_sum / batch_count


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.PT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
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
        total_loss, policy_loss, value_loss = train_epoch(
            model, train_dataloader, optimizer, hp.DEVICE, estimated_batches
        )
        scheduler.step()
        
        print(f"Train - Loss: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")
        
        # Validate every epoch
        val_loss, val_policy_loss, val_value_loss = validate(model, val_dataloader, hp.DEVICE)
        print(f"Valid - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), hp.PT_MODEL_PATH)
    print(f"\nTraining complete! Model saved at '{hp.PT_MODEL_PATH}'")


if __name__ == "__main__":
    main()
