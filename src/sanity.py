"""A suite of sanity tests to ensure basic functionality of the FastChess components."""
import chess
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from datasets import load_dataset

from tqdm import tqdm

import hyperparams as hp
from model import FastChessNet
from pretrain import ChessValidationDataset, ChessStreamDataset, ChessPuzzleDataset, MixedDataLoader
from mcts import MCTS, get_best_move
from selfplay import SelfPlayDataset
from util import MoveEncoder, ChessPositionEncoder, print_board

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

move_encoder = MoveEncoder()
board_encoder = ChessPositionEncoder()

if hp.USE_SEED:
    torch.manual_seed(hp.SEED)

# ***************
# BASIC TESTS
# ***************

def encode(board):
    encoder = ChessPositionEncoder()
    return torch.tensor(encoder.encode_board(board), dtype=torch.float32).unsqueeze(0).to(hp.DEVICE)


def make_policy_target(move):
    pi = torch.zeros(8 * 8 * 73, device=hp.DEVICE)
    encoder = MoveEncoder()
    idx = encoder.encode_move(move)
    pi[idx] = 1.0
    return pi.unsqueeze(0)


def test_move_encoding():
    """Test that move encoding and decoding are consistent."""
    all_moves = []

    board = chess.Board()
    for move in board.legal_moves:
        print("Legal mask shape:", move_encoder.get_legal_mask(board).shape)
        all_moves.append(move)

    print("[Move Encoding Test]")
    success = True
    for move in all_moves:
        idx = move_encoder.encode_move(move)
        decoded_move = move_encoder.decode_move(idx)
        if move != decoded_move:
            print(f"Mismatch: original {move}, decoded {decoded_move}")
            success = False

    if success:
        print("All moves encoded and decoded correctly.")
    else:
        print("Some moves failed encoding/decoding test.")

def test_board_encoding():
    """Test that board encoding produces the expected shape and values."""
    board = chess.Board()
    print("Board before encoding: ")
    print_board(board)
    encoded = board_encoder.encode_board(board)
    board = board_encoder.decode_board(encoded)
    print("Board after encoding and decoding: ")
    print_board(board)
    

def test_forward_sanity(net):
    """Test that the network's value output changes sign when the board is mirrored (only relevant after training)."""
    board = chess.Board()
    mirror = board.mirror()

    with torch.no_grad():
        _, v1 = net(encode(board), torch.tensor(move_encoder.get_legal_mask(board)).unsqueeze(0).to(hp.DEVICE))
        _, v2 = net(encode(mirror), torch.tensor(move_encoder.get_legal_mask(mirror)).unsqueeze(0).to(hp.DEVICE))

    print("[Forward sanity]")
    print("Value original:", v1.item())
    print("Value mirror  :", v2.item())

    if v1.item() * v2.item() >= 0:
        print("Perspective bug")
    else:
        print("Perspective OK")


def test_overfit_single_position(net):
    """Test that the network can overfit a single position."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    pi_target = make_policy_target(board, move)
    z_target = torch.tensor([[0.1]], device=DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for i in range(300):
        p, v = net(encode(board), torch.tensor(move_encoder.get_legal_mask(board)).unsqueeze(0).to(hp.DEVICE))
        loss_p = F.cross_entropy(p, pi_target.argmax(dim=1))
        loss_v = F.mse_loss(v, z_target)
        loss = loss_p + loss_v

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print(f"Step {i}, Policy Loss: {loss_p.item():.4f}, Value Loss: {loss_v.item():.4f}")

    with torch.no_grad():
        p, v = net(encode(board), torch.tensor(move_encoder.get_legal_mask(board)).unsqueeze(0).to(hp.DEVICE))
        prob = F.softmax(p, dim=1)[0, pi_target.argmax()]

    print("[Overfit single position]")
    print("Policy prob:", prob.item())
    print("Value:", v.item())

    if prob > 0.99:
        print("Can overfit one position")
    else:
        print("Cannot overfit one position")


def test_overfit_tiny_game(net):
    """Test that the network can overfit a tiny game sequence."""
    board = chess.Board()
    game = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]

    samples = []
    for move in game:
        pi = make_policy_target(board, move)
        samples.append((deepcopy(board), pi))
        board.push(move)

    z = torch.tensor([[0.2]], device=DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    for i in range(400):
        for b, pi in samples:
            p, v = net(encode(b), torch.tensor(move_encoder.get_legal_mask(b)).unsqueeze(0).to(hp.DEVICE))
            loss = F.cross_entropy(p, pi.argmax(dim=1)) + F.mse_loss(v, z)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                with torch.no_grad():
                    p, v = net(encode(b), torch.tensor(move_encoder.get_legal_mask(b)).unsqueeze(0).to(hp.DEVICE))
                    loss_p = F.cross_entropy(p, pi.argmax(dim=1))
                    loss_v = F.mse_loss(v, z)
                    print(f"Step {i}, Policy Loss: {loss_p.item():.4f}, Value Loss: {loss_v.item():.4f}")

    print("[Overfit tiny game] (if loss decreased smoothly)")

# ***************
# PRETRAINING TESTS
# ***************

def examine_pt_dataset(net, test_shapes=True, examine_moves=True, test_batching=True):
    """Examine the pretraining datasets and dataloaders to ensure they are loading and batching data correctly."""
    # Temporary hyperparameters for dataset sizes (these would be set in hyperparams.py in a real run)
    PT_STEPS = 10
    PT_BATCH_SIZE = 256
    TACTICAL_RATIO = 0.1
    test_train_positional_positions =  (PT_BATCH_SIZE - int(PT_BATCH_SIZE * TACTICAL_RATIO)) * PT_STEPS
    test_val_positional_positions = int(0.01 * test_train_positional_positions)
    test_train_puzzle_positions = int(PT_BATCH_SIZE * TACTICAL_RATIO) * PT_STEPS
    test_val_puzzle_positions = int(0.01 * test_train_puzzle_positions)

    # Load validation positional data dataset from HuggingFace (training data is streamed in pretrain_epoch)
    val_positional_dataset_hf = load_dataset(
        "angeluriot/chess_games",
        split="train",
        streaming=True
    )
    
    print(f"\nPositional dataset: {test_train_positional_positions:,} training positions, {test_val_positional_positions:,} validation positions")
    print(f"Puzzle dataset: {test_train_puzzle_positions:,} training positions, {test_val_puzzle_positions:,} validation positions")
    
    # Create fixed positional validation set (load into memory)
    val_positional_dataset = ChessValidationDataset(
        hf_dataset=val_positional_dataset_hf,
        num_positions=test_val_positional_positions,
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
        num_positions=test_train_puzzle_positions,
        skip_positions=test_val_puzzle_positions,
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
        num_positions=test_val_puzzle_positions,
        skip_positions=0,
        min_rating=hp.MIN_PUZZLE_RATING
    )
    
    # Calculate batches per epoch based on positional data (puzzles are sampled in each batch)
    estimated_batches = int(test_train_positional_positions / (hp.PT_BATCH_SIZE * (1 - hp.TACTICAL_RATIO)))
    
    # Create mixed validation dataloader
    val_dataloader = MixedDataLoader(
        positional_dataset=val_positional_dataset,
        puzzle_dataset=val_puzzle_dataset,
        batch_size=hp.PT_BATCH_SIZE,
        tactical_ratio=hp.TACTICAL_RATIO
    )
    
    # Calculate validation batches
    val_batches = int(len(val_positional_dataset) / (hp.PT_BATCH_SIZE * (1 - hp.TACTICAL_RATIO)))
    
    print(f"\nTraining set: {test_train_positional_positions:,} positional + {len(train_puzzle_dataset):,} tactical positions in ~{estimated_batches:,} batches per epoch")
    print(f"Validation set: {len(val_positional_dataset):,} positional + {len(val_puzzle_dataset):,} tactical positions in {val_batches:,} batches")
    print(f"Batch composition: {int((1-hp.TACTICAL_RATIO)*100)}% games, {int(hp.TACTICAL_RATIO*100)}% puzzles")

    # Create streaming dataset like I would each epoch
    positional_dataset = load_dataset(
        "angeluriot/chess_games",
        split="train",
        streaming=True
    )
    train_dataset = ChessStreamDataset(
        hf_dataset=positional_dataset,
        min_elo=hp.MIN_ELO,
        max_positions=test_train_positional_positions,
        skip_positions=test_val_positional_positions
    )
    
    # Create mixed dataloader
    mixed_dataloader = MixedDataLoader(
        positional_dataset=train_dataset,
        puzzle_dataset=train_puzzle_dataset,
        batch_size=hp.PT_BATCH_SIZE,
        tactical_ratio=hp.TACTICAL_RATIO
    )

    pbar = tqdm(mixed_dataloader, total=estimated_batches)
    for i, (states, target_moves, legal_masks, target_values) in enumerate(pbar):
        if i >= 3 and not test_batching: 
            break
        states = states.to(DEVICE)
        target_moves = target_moves.to(DEVICE).squeeze(1)
        legal_masks = legal_masks.to(DEVICE)
        target_values = target_values.to(DEVICE).squeeze(1)
        policy_logits, value_pred = net(states, legal_masks)
        print(f"\nBatch {i+1}")
        if test_shapes:
            print("State shape:", states.shape)
            print("Mask shape:", legal_masks.shape)
            print("Policy shape:", target_moves.shape)
            print("Value shape:", target_values.shape)
            print("Policy logits shape:", policy_logits.shape)
            print("Value pred shape:", value_pred.shape)

        if examine_moves:
            print_board(board_encoder.decode_board(states[0].cpu().numpy()))
            print("Ground truth move:", move_encoder.decode_move(target_moves[0].item()))
            print("Ground truth value:", target_values[0].item())


# ***************
# SELF PLAY TESTS
# ***************


class DummyNet(torch.nn.Module):
    """A dummy network that outputs uniform policy and zero value."""
    def forward(self, x, legal_mask):
        batch = x.size(0)
        p = torch.ones(batch, hp.POLICY_DIM, device=hp.DEVICE)
        v = torch.zeros(batch, 1, device=hp.DEVICE)
        return p, v


def test_mcts_dummy():
    """Ensure MCTS can still run with a dummy network."""
    board = chess.Board()
    net = DummyNet().to(hp.DEVICE)
    mcts = MCTS(net, c_puct=1.25)

    pi = mcts.search(board, sims=50)

    print("[MCTS dummy net]")
    print("Moves:", list(pi.keys())[:5])

    if all(board.is_legal(m) for m in pi):
        print("MCTS works without NN")
    else:
        print("Illegal moves from MCTS")


def test_mate_in_1(net):
    """Test that MCTS can find a mate-in-1 position."""
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R6K w - - 0 1")

    mcts = MCTS(net, c_puct=1.25)
    pi = mcts.search(board, sims=400)
    for move, visits in pi.items():
        print(
            f"Move: {move.uci()}, ",
            f"Visits: {visits}, ",
            f"-Q: {-1*mcts.root.children[move].Q:.4f}, ",
            f"Prior: {mcts.root.children[move].P:.4f}, ",
            f"UCB: {mcts.c_puct * mcts.root.children[move].P * (mcts.root.N ** 0.5) / (1 + mcts.root.children[move].N):.4f}"
        )

    best = max(pi.items(), key=lambda x: x[1])[0]
    best = get_best_move(pi, temperature=0.0)

    print("[Mate-in-1]")
    print("Predicted best move:", best)

    board.push(best)
    if board.is_checkmate():
        print("Mate found")
    else:
        print("Mate missed")


def test_replay_buffer(net):
    """Play a full game and store positions in the replay buffer, then sample from it."""
    buffer = SelfPlayDataset(net)
    print(len(buffer))  # Should be 0
    board = chess.Board()
    mcts = MCTS(net, c_puct=hp.CPUCT)
    data = []

    while not board.is_game_over():
        policy_dict = mcts.search(board, hp.SIMULATIONS_TRAIN)

        # Build full policy vector
        policy = np.zeros(8*8*73, dtype=np.float32)
        for move, visits in policy_dict.items():
            idx = move_encoder.encode_move(move)
            policy[idx] = visits

        policy /= (policy.sum() + 1e-8)

        data.append((board_encoder.encode_board(board), move_encoder.get_legal_mask(board), policy))
        temperature = hp.TEMPERATURE if board.fullmove_number < hp.TEMPERATURE_MOVES else 0.0
        move = get_best_move(policy_dict, temperature)
        board.push(move)
        print_board(board) #DEBUG

    value = 0
    if board.result() == "1-0": value = 1
    elif board.result() == "0-1": value = -1

    for state, mask, policy in data:
        buffer.push(state, mask, policy, value)

    print("[Replay Buffer]")
    print("Buffer size:", len(buffer))
    sample_state, sample_mask, sample_policy, sample_value = buffer.sample()
    print("Sampled state shape:", sample_state.shape)
    print("Sampled mask shape:", sample_mask.shape)
    print("Sampled policy shape:", sample_policy.shape)
    print("Sampled value shape:", sample_value.shape)


def observe_selfplay_game(net):
    """Play a self-play game and display the board after each move."""
    dataset = SelfPlayDataset(net)
    dataset.self_play(display=True)

if __name__ == "__main__":
    # Uncomment to use default vs trained model

    # net = DummyNet().to(DEVICE)
    net = FastChessNet().to(DEVICE)
    # net.load_state_dict(torch.load(hp.PT_MODEL_PATH, map_location=DEVICE))   
    # net.load_state_dict(torch.load("../models/checkpoint_3.pt", map_location=DEVICE))   
    # net.load_state_dict(torch.load(hp.MODEL_PATH, map_location=DEVICE))   

    # Uncomment to run individual tests

    # BASIC TESTS
    # test_move_encoding()
    # test_board_encoding()
    # test_forward_sanity(net)
    # test_overfit_single_position(net)
    # test_overfit_tiny_game(net)

    # PRETRAINING TESTS
    examine_pt_dataset(net, False, False)

    # MCTS AND SELF PLAY TESTS
    # test_mcts_dummy()
    # test_mate_in_1(net)
    # test_replay_buffer(net)
    # observe_selfplay_game(net)
