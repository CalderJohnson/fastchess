"""Training scheme for FastChessNet using self-play."""
import torch
import torch.optim as optim
import numpy as np
import chess

import hyperparams as hp
from model import FastChessNet
from mcts import MCTS, get_best_move
from util import encode_board, move_to_index, print_board, get_legal_mask
from replay import ReplayBuffer

if hp.USE_SEED:
    torch.manual_seed(hp.SEED)

def self_play(net):
    """Plays a single game using MCTS and returns training data."""
    board = chess.Board()
    mcts = MCTS(net, c_puct=hp.CPUCT)
    data = []

    while not board.is_game_over():
        policy_dict = mcts.search(board, hp.SIMULATIONS_TRAIN)

        # Build full policy vector
        policy = np.zeros(8*8*73, dtype=np.float32)
        for move, visits in policy_dict.items():
            fr, fc, p = move_to_index(move)
            idx = fr*8*73 + fc*73 + p
            policy[idx] = visits

        policy /= (policy.sum() + 1e-8)

        data.append((encode_board(board), get_legal_mask(board), policy))
        temperature = hp.TEMPERATURE if board.fullmove_number < hp.TEMPERATURE_MOVES else 0.0
        move = get_best_move(policy_dict, temperature)
        board.push(move)
        print_board(board) #DEBUG

    value = 0
    if board.result() == "1-0": value = 1
    elif board.result() == "0-1": value = -1

    return [(state, mask, policy, value) for state, mask, policy in data]

def train(net, replay_buffer):
    """Trains the network using data from the replay buffer."""
    optimizer = optim.Adam(net.parameters(), lr=hp.LR)
    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.MSELoss()

    for _ in range(hp.EPOCHS):
        state, mask, policy, value = replay_buffer.sample()
        state = torch.tensor(state).to(hp.DEVICE)
        mask = torch.tensor(mask).to(hp.DEVICE)
        policy = torch.tensor(policy).to(hp.DEVICE)
        value = torch.tensor(value).float().to(hp.DEVICE)

        out_p, out_v = net(state, mask)

        loss_v = criterion_value(out_v.squeeze(), value)
        loss_p = criterion_policy(out_p, policy)
        loss = loss_v + loss_p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def main():
    """Main training loop."""
    net = FastChessNet().to(hp.DEVICE)
    buffer = ReplayBuffer()

    print("Starting training...")

    for i in range(hp.ITERATIONS):
        for _ in range(hp.GAMES_PER_ITER):
            for sample in self_play(net):
                buffer.push(*sample)

        if len(buffer.buffer) >= hp.BATCH_SIZE:
            train(net, buffer)

        print("Iteration", i, "complete")

    torch.save(net.state_dict(), hp.MODEL_PATH)

if __name__ == "__main__":
    main()
