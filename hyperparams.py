"""Global configs"""
import torch

# Misc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_SEED = True
SEED = 42
MODEL_PATH = "./models/model.pt"
REPLAY_CACHE_PATH = "./replay_cache/"

# Model hyperparameters
N_RES_BLOCKS = 10   # Number of stacked residual blocks
N_CHANNELS   = 128  # Number of channels in convolutional layers
POLICY_DIM   = 4672 # Dimension of the policy output
P_FC_DIM     = 2048 # Fully connected layer dimension for policy head
V_FC1_DIM    = 192  # First fully connected layer dimension for value head
V_FC2_DIM    = 128  # Second fully connected layer dimension for value head

# Training hyperparameters
BATCH_SIZE = 64        # Samples per training batch
EPOCHS = 2             # Number of training epochs (over replay buffer) per iteration
ITERATIONS = 20        # Total training iterations (self-play followed by training)
REPLAY_SIZE = 2000     # Size of the replay buffer (number of position, move, outcome tuples)
LR = 1e-3              # Initial learning rate
GAMES_PER_ITER = 1     # Self-play games per training iteration
TEMPERATURE_MOVES = 10 # Number of moves with exploration in self-play
TEMPERATURE = 1.0      # Temperature for move selection

# MCTS hyperparameters
SIMULATIONS_TRAIN = 100 # Number of MCTS rollouts per move during training
SIMULATIONS_INF = 100   # Number of MCTS rollouts per move during inference
CPUCT = 1.25            # Exploration constant
