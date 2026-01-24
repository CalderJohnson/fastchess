"""A suite of sanity tests to ensure basic functionality of the FastChess components."""
import chess
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

import hyperparams as hp
from model import FastChessNet
from mcts import MCTS, get_best_move
from selfplay import SelfPlayDataset
from util import MoveEncoder, ChessPositionEncoder, print_board

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

move_encoder = MoveEncoder()
board_encoder = ChessPositionEncoder()

if hp.USE_SEED:
    torch.manual_seed(hp.SEED)

def encode(board):
    encoder = ChessPositionEncoder()
    return torch.tensor(encoder.encode_board(board), dtype=torch.float32).unsqueeze(0).to(DEVICE)


def make_policy_target(move):
    pi = torch.zeros(8 * 8 * 73, device=DEVICE)
    encoder = MoveEncoder()
    idx = encoder.encode_move(move)
    pi[idx] = 1.0
    return pi.unsqueeze(0)


def test_move_encoding():
    """Test that move encoding and decoding are consistent."""
    all_moves = []

    board = chess.Board()
    for move in board.legal_moves:
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


def test_forward_sanity(net):
    """Test that the network's value output changes sign when the board is mirrored (only relevant after training)."""
    board = chess.Board()
    mirror = board.mirror()

    with torch.no_grad():
        _, v1 = net(encode(board), torch.tensor(move_encoder.get_legal_mask(board)).unsqueeze(0).to(DEVICE))
        _, v2 = net(encode(mirror), torch.tensor(move_encoder.get_legal_mask(mirror)).unsqueeze(0).to(DEVICE))

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
        p, v = net(encode(board), torch.tensor(move_encoder.get_legal_mask(board)).unsqueeze(0).to(DEVICE))
        loss_p = F.cross_entropy(p, pi_target.argmax(dim=1))
        loss_v = F.mse_loss(v, z_target)
        loss = loss_p + loss_v

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print(f"Step {i}, Policy Loss: {loss_p.item():.4f}, Value Loss: {loss_v.item():.4f}")

    with torch.no_grad():
        p, v = net(encode(board), torch.tensor(move_encoder.get_legal_mask(board)).unsqueeze(0).to(DEVICE))
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
            p, v = net(encode(b), torch.tensor(move_encoder.get_legal_mask(b)).unsqueeze(0).to(DEVICE))
            loss = F.cross_entropy(p, pi.argmax(dim=1)) + F.mse_loss(v, z)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                with torch.no_grad():
                    p, v = net(encode(b), torch.tensor(move_encoder.get_legal_mask(b)).unsqueeze(0).to(DEVICE))
                    loss_p = F.cross_entropy(p, pi.argmax(dim=1))
                    loss_v = F.mse_loss(v, z)
                    print(f"Step {i}, Policy Loss: {loss_p.item():.4f}, Value Loss: {loss_v.item():.4f}")

    print("[Overfit tiny game] (if loss decreased smoothly)")


class DummyNet(torch.nn.Module):
    """A dummy network that outputs uniform policy and zero value."""
    def forward(self, x, legal_mask):
        batch = x.size(0)
        p = torch.ones(batch, 8 * 8 * 73, device=x.device)
        v = torch.zeros(batch, 1, device=x.device)
        return p, v


def test_mcts_dummy():
    """Ensure MCTS can still run with a dummy network."""
    board = chess.Board()
    net = DummyNet().to(DEVICE)
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
    pi = mcts.search(board, sims=100, add_noise=False)
    for move, visits in pi.items():
        print(f"Move: {move.uci()}, Visits: {visits}")

    best = max(pi.items(), key=lambda x: x[1])[0]

    print("[Mate-in-1]")
    print("Best move:", best)

    board.push(best)
    if board.is_checkmate():
        print("Mate found")
    else:
        print("Mate missed")


def test_replay_buffer(net):
    """Play a full game and store positions in the replay buffer, then sample from it."""
    buffer = SelfPlayDataset()
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

if __name__ == "__main__":
    # Uncomment to use default vs trained model

    net = FastChessNet().to(DEVICE)
    # net.load_state_dict(torch.load(hp.MODEL_PATH, map_location=DEVICE))   

    # Uncomment to run individual tests

    test_move_encoding()
    # test_forward_sanity(net)
    # test_overfit_single_position(net)
    # test_overfit_tiny_game(net)
    # test_mcts_dummy()
    # test_mate_in_1(net)
    # test_replay_buffer(net)
