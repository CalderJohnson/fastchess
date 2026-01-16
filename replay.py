"""Replay buffer to store self-play data"""
import random
import numpy as np
from collections import deque
import hyperparams as hp

class ReplayBuffer:
    """Fixed-size buffer to store self-play data."""
    def __init__(self):
        self.buffer = deque(maxlen=hp.REPLAY_SIZE)

    def push(self, state, mask, choices, result):
        self.buffer.append((state, mask, choices, result))

    def sample(self):
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
        self.buffer = deque(data, maxlen=hp.REPLAY_SIZE)
    
    def __len__(self):
        return len(self.buffer)
