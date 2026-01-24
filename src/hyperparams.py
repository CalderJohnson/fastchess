"""Global configs"""
import torch

# Misc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SEED = True
SEED = 42
MODEL_PATH = "../models/model.pt"
PT_MODEL_PATH = "../models/pretrained_model.pt"
PT_DATASET_PATH = "../data/pretrain/dataset/"
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
N_GAMES = 500000             # Number of games to sample for pretraining
MAX_POSITIONS_PER_GAME = 40  # Max positions sampled per game
CHUNK_SIZE = 10000           # Number of games per saved chunk
PT_BATCH_SIZE = 256          # Batch size for pretraining
PT_LR = 1e-3                 # Learning rate for pretraining
PT_EPOCHS = 10               # Number of pretraining epochs
PT_SPLIT = 0.95              # Train/validation split ratio for pretraining

# Self-play training hyperparameters
BATCH_SIZE = 64        # Samples per training batch
TRAINING_STEPS = 128   # Number of training steps (over replay buffer) per iteration
ITERATIONS = 50        # Total training iterations (self-play followed by training)
REPLAY_SIZE = 2000     # Size of the replay buffer (number of position, move, outcome tuples)
LR = 1e-3              # Initial learning rate
GAMES_PER_ITER = 2     # Self-play games per training iteration
TEMPERATURE_MOVES = 10 # Number of moves with exploration in self-play
TEMPERATURE = 1.0      # Temperature for move selection

# MCTS hyperparameters
SIMULATIONS_TRAIN = 100 # Number of MCTS rollouts per move during training
SIMULATIONS_INF = 100   # Number of MCTS rollouts per move during inference
CPUCT = 1.25            # Exploration constant
