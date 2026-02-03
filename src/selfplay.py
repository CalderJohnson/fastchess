"""Replay buffer to store self-play data"""
import torch
import torch.nn.functional as F
import random
import chess
import numpy as np
from collections import deque
import hyperparams as hp
from mcts import MCTS, get_best_move
from util import MoveEncoder, ChessPositionEncoder, print_board


class _ReplayBuffer:
    """Fixed-size buffer to store self-play data."""
    def __init__(self):
        self.buffer = deque(maxlen=hp.REPLAY_SIZE)
        self.move_encoder = MoveEncoder()
        self.board_encoder = ChessPositionEncoder()

    def push(self, state, mask, choices, result):
        """Add a new experience to the buffer."""
        self.buffer.append((state, mask, choices, result))

    def sample(self):
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, hp.BATCH_SIZE)
        state, mask, choices, result = zip(*batch)
        return np.array(state), np.array(mask), np.array(choices), np.array(result)
    
    def clear(self):
        """Clears the replay buffer."""
        self.buffer.clear()

    def save(self, filename):
        """Saves the replay buffer to a file."""
        np.save(filename, list(self.buffer))

    def load(self, filename):
        """Loads the replay buffer from a file."""
        data = np.load(filename, allow_pickle=True)
        self.buffer = deque(data, maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)
    

class SelfPlayDataset(_ReplayBuffer):
    """Dataset wrapper for self-play buffer."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.board_encoder = ChessPositionEncoder()
        self.move_encoder = MoveEncoder()
    
    def self_play(self, display=False):
        """Plays a single game using MCTS and saves training data."""
        self.net.eval()
        board = chess.Board()
        mcts = MCTS(self.net, c_puct=hp.CPUCT)
        data = []

        while not board.is_game_over():
            policy_dict = mcts.search(board, hp.SIMULATIONS_TRAIN)

            # Build full policy vector
            policy = np.zeros(8*8*73, dtype=np.float32)
            for move, visits in policy_dict.items():
                idx = self.move_encoder.encode_move(move)
                policy[idx] = visits

            policy /= (policy.sum() + 1e-8)

            data.append((self.board_encoder.encode_board(board), self.move_encoder.get_legal_mask(board), policy))
            temperature = hp.TEMPERATURE if board.fullmove_number < hp.TEMPERATURE_MOVES else 0.0
            move = get_best_move(policy_dict, temperature)
            board.push(move)
            if display:
                print_board(board) #DEBUG

        value = 0
        if board.result() == "1-0": value = 1
        elif board.result() == "0-1": value = -1

        for state, mask, policy in data:
            self.push(state, mask, policy, value)


def selfplay_train_iteration(model, replay_buffer, optimizer):
    """Trains the network using data from the replay buffer."""
    model.train()
    for _ in range(hp.TRAINING_STEPS):
        state, mask, policy, value = replay_buffer.sample()
        state = torch.tensor(state).to(hp.DEVICE)
        mask = torch.tensor(mask).to(hp.DEVICE)
        policy = torch.tensor(policy).to(hp.DEVICE)
        value = torch.tensor(value).float().to(hp.DEVICE)

        out_p, out_v = model(state, mask)

        loss_v = F.mse_loss(out_v.squeeze(), value)
        loss_p = F.cross_entropy(out_p, policy.argmax(dim=1))
        loss = loss_v + loss_p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
