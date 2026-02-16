"""Pretraining pipeline for FastChessNet using historical chess games and puzzles."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
import chess
from tqdm import tqdm
from util import ChessPositionEncoder, MoveEncoder


class ChessPuzzleDataset(Dataset):
    """Dataset of chess puzzles for tactical training."""
    
    def __init__(self, hf_dataset, num_puzzles, skip_puzzles=0, min_rating=1500):
        """
        Args:
            hf_dataset: HuggingFace puzzle dataset
            num_puzzles: Number of puzzles to load
            min_rating: Minimum puzzle rating to include
            skip_puzzles: Number of puzzles to skip (for validation split)
        """
        self.encoder = ChessPositionEncoder()
        self.move_encoder = MoveEncoder()
        self.positions = []
        
        print(f"Loading {num_puzzles} chess puzzles...")
        puzzle_count = 0
        
        for puzzle in tqdm(hf_dataset.skip(skip_puzzles), desc="Loading puzzles", total=num_puzzles):
            if puzzle_count >= num_puzzles:
                break
            
            # Filter by rating
            if puzzle['Rating'] < min_rating:
                continue
            
            # Parse puzzle
            try:
                board = chess.Board(puzzle['FEN'])
                moves = puzzle['Moves'].split()
                
                # Process each move in the solution
                # Puzzles show the best continuation, so each move is "correct"
                for i in range(0, len(moves), 2):  # Only use player's moves (every other move)
                    if i >= len(moves):
                        break
                    
                    move_uci = moves[i]
                    move = chess.Move.from_uci(move_uci)
                    
                    if move not in board.legal_moves:
                        break
                    
                    # For puzzles, assume the solution leads to winning
                    # Value = +1 for side to move (they're solving the puzzle)
                    value = 1.0
                    
                    move_idx = self.move_encoder.encode_move(move)
                    
                    self.positions.append({
                        'fen': board.fen(),
                        'move_idx': move_idx,
                        'value': value
                    })
                    
                    # Make the move and opponent's response
                    board.push(move)
                    if i + 1 < len(moves):
                        opponent_move = chess.Move.from_uci(moves[i + 1])
                        if opponent_move in board.legal_moves:
                            board.push(opponent_move)
                        else:
                            break
                
                puzzle_count += 1
                
            except Exception as e:
                continue
        
        print(f"Loaded {len(self.positions)} tactical positions from {puzzle_count} puzzles")
    
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


class MixedDataLoader:
    """DataLoader that yields mixed batches from positional game data and tactical puzzle data."""
    
    def __init__(self, positional_dataset, puzzle_dataset, batch_size, tactical_ratio=0.2):
        """
        Args:
            positional_dataset: Dataset for game positions (streaming or fixed)
            puzzle_dataset: Fixed puzzle dataset
            batch_size: Total batch size
            tactical_ratio: Fraction of batch from puzzles (0.2 = 20%)
        """
        self.positional_dataset = positional_dataset
        self.puzzle_dataset = puzzle_dataset
        self.batch_size = batch_size
        self.tactical_size = int(batch_size * tactical_ratio)
        self.positional_size = batch_size - self.tactical_size
        self.positional_iter = None
        
        # Check if positional_dataset is a fixed Dataset (has __getitem__) or streaming
        self.is_fixed_positional = hasattr(positional_dataset, '__getitem__') and hasattr(positional_dataset, '__len__')
        
        # For fixed datasets, track position in shuffled indices
        self.positional_indices = None
        self.positional_index = 0
    
    def __iter__(self):
        """Yield mixed batches."""
        if self.is_fixed_positional:
            # Shuffle indices once at the start
            self.positional_indices = np.random.permutation(len(self.positional_dataset))
            self.positional_index = 0
        else:
            self.positional_iter = iter(self.positional_dataset)
        return self
    
    def __next__(self):
        """Get next mixed batch."""
        # Accumulate positional data
        states_list = []
        moves_list = []
        masks_list = []
        values_list = []
        
        # Get positional data (either from stream or fixed dataset)
        if self.is_fixed_positional:
            # Iterate through shuffled indices without replacement
            for _ in range(self.positional_size):
                if self.positional_index >= len(self.positional_indices):
                    raise StopIteration
                
                idx = self.positional_indices[self.positional_index]
                self.positional_index += 1
                
                state, move, mask, value = self.positional_dataset[idx]
                states_list.append(state)
                moves_list.append(move)
                masks_list.append(mask)
                values_list.append(value)
        else:
            # Get from streaming dataset
            for _ in range(self.positional_size):
                try:
                    state, move, mask, value = next(self.positional_iter)
                    states_list.append(state)
                    moves_list.append(move)
                    masks_list.append(mask)
                    values_list.append(value)
                except StopIteration:
                    raise StopIteration
        
        # Sample tactical data from puzzles
        if len(self.puzzle_dataset) > 0:
            puzzle_indices = np.random.choice(len(self.puzzle_dataset), 
                                            self.tactical_size, replace=True)
            
            for idx in puzzle_indices:
                state, move, mask, value = self.puzzle_dataset[idx]
                states_list.append(state)
                moves_list.append(move)
                masks_list.append(mask)
                values_list.append(value)
        
        # Stack into batch tensors
        return (
            torch.stack(states_list),
            torch.stack(moves_list),
            torch.stack(masks_list),
            torch.stack(values_list)
        )


def pretrain_epoch(model, dataloader, optimizer, scheduler, device, total_batches=None):
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
        scheduler.step()
        
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


def validate(model, dataloader, device, total_batches=None):
    """Run validation."""
    model.eval()
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    batch_count = 0
    
    with torch.no_grad():
        for states, target_moves, legal_masks, target_values in tqdm(dataloader, desc="Validating", total=total_batches):
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
