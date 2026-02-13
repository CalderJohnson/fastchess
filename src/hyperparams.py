"""Global configs"""
import torch
import math

# Misc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SEED = True
SEED = 42
MODEL_PATH = "../models/"
PT_MODEL_PATH = "../models/pretrained_model.pt"
REPLAY_CACHE_PATH = "../data/cache/"

# Model hyperparameters
N_RES_BLOCKS = 10   # Number of stacked residual blocks
N_CHANNELS   = 128  # Number of channels in convolutional layers
POLICY_DIM   = 4672 # Dimension of the policy output
P_FC_DIM     = 2048 # Fully connected layer dimension for policy head
V_FC1_DIM    = 192  # First fully connected layer dimension for value head
V_FC2_DIM    = 128  # Second fully connected layer dimension for value head

# Pretraining hyperparameters
MIN_ELO = 2000               # Minimum Elo rating for games used in pretraining
N_GAMES = 100000             # Number of games to sample for pretraining
MAX_POSITIONS_PER_GAME = 40  # Max positions sampled per game
N_PUZZLES = 100000           # Number of puzzles to load
MIN_PUZZLE_RATING = 1000     # Minimum puzzle difficulty
TACTICAL_RATIO = 0.1         # 10% of each batch from puzzles
PT_BATCH_SIZE = 256          # Batch size for pretraining
PT_LR = 1e-3                 # Learning rate for pretraining
PT_EPOCHS = 10               # Number of pretraining epochs
PT_SPLIT = 0.99              # Train/validation split ratio for pretraining

# Self-play training hyperparameters
BATCH_SIZE = 128        # Samples per training batch
TRAINING_STEPS = 500   # Number of training steps (over replay buffer) per iteration
ITERATIONS = 4         # Total training iterations (self-play followed by training) per epoch
REPLAY_SIZE = 5000     # Size of the replay buffer (number of position, move, outcome tuples)
LR = 1e-4              # Initial learning rate
GAMES_PER_ITER = 8     # Self-play games per training iteration
TEMPERATURE_MOVES = 10 # Number of moves with exploration in self-play
TEMPERATURE = 1.0      # Temperature for move selection

# MCTS hyperparameters
SIMULATIONS_TRAIN = 400 # Number of MCTS rollouts per move during training
SIMULATIONS_INF = 400   # Number of MCTS rollouts per move during inference
CPUCT = math.sqrt(2)    # Exploration constant
