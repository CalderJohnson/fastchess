The most efficient chess engine I can make through self play and deep RL.

We begin with a pretraining phase using: https://huggingface.co/datasets/angeluriot/chess_games

The chess engine trains itself further using a modified version of the Monte Carlo Search Tree.

Our MCTS works by:

Selection: Traverse the tree using the computed values of the nodes picked from a normal distribution until a leaf is reached.
Expansion: Expand the leaf node by creating a child for every possible action.
Evaluation: The value of the leaf node is estimated using our model.
Backpropagation: Backpropagate the estimation to the root node.
This is repeated X times (X being the depth setting).

After a number of games of self play, the model replays its actions using the updated value/policy estimates discovered by MCTS. It learns the updated value/policy, improving itself incrementally through each iteration.

The model itself has the following architecture:

Input layer: 18 8x8 bitboards representing game state.

Hidden layers: 1 convolutional layer, followed by 10 residual blocks with skip connections.

Output: The value of of the given board (scalar, -1 to 1), and a weighted vector of possible moves.
