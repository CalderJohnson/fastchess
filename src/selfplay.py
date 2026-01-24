"""Replay buffer to store self-play data"""
import random
import chess
import numpy as np
from collections import deque
import hyperparams as hp
from mcts import MCTS, get_best_move
from util import MoveEncoder, ChessPositionEncoder


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
    

class SelfPlayDataset(_ReplayBuffer):
    """Dataset wrapper for self-play buffer."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.board_encoder = ChessPositionEncoder()
        self.move_encoder = MoveEncoder()
    
    def self_play(self):
        """Plays a single game using MCTS and saves training data."""
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
            # print_board(board) #DEBUG

        value = 0
        if board.result() == "1-0": value = 1
        elif board.result() == "0-1": value = -1

        for state, mask, policy in data:
            self.push(state, mask, policy, value)
