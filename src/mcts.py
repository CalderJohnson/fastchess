"""Monte Carlo Tree Search implementation for FastChess."""
import torch
import torch.nn.functional as F
import math
import numpy as np
from util import MoveEncoder, ChessPositionEncoder

class Node:
    """Individual node in the MCTS tree."""
    def __init__(self, prior):
        self.N = 0            # Visit count
        self.W = 0.0          # Accumulated value
        self.Q = 0.0          # Mean value
        self.P = prior        # Prior probability
        self.children = {}    # Move -> Node
        self.expanded = False # Whether node is expanded

class MCTS:
    """Monte Carlo Tree Search implementation."""
    def __init__(self, net, c_puct):
        self.net = net
        self.c_puct = c_puct
        self.move_encoder = MoveEncoder()
        self.board_encoder = ChessPositionEncoder()

    def _select(self, node):
        best_score = -float('inf')
        best_move, best_child = None, None
        sqrt_N = math.sqrt(node.N + 1)

        for move, child in node.children.items():
            u = self.c_puct * child.P * sqrt_N / (1 + child.N)
            score = (-1 * child.Q) + u
            if score > best_score:
                best_score = score
                best_move, best_child = move, child

        return best_move, best_child

    def _expand(self, node, board):
        """From a leaf node, expand the tree by adding all legal moves as children."""

        # Encode the board and mask legal moves
        encoded = torch.tensor(self.board_encoder.encode_board(board)).unsqueeze(0).cuda()
        moves = list(board.legal_moves)
        legal_mask = self.move_encoder.get_legal_mask(board)
        legal_mask = torch.tensor(legal_mask).unsqueeze(0).cuda()

        # Evaluate with the neural network
        with torch.no_grad():
            policy, value = self.net(encoded, legal_mask)
            probs = F.softmax(policy, dim=1).cpu().numpy()[0]
            value = value.item()

        # Create child nodes with prior probabilities
        priors = []
        for move in moves:
            idx = self.move_encoder.encode_move(move)
            priors.append(max(probs[idx], 1e-8))  # Floor to avoid zero probability
        
        priors = np.array(priors)
        priors /= (priors.sum() + 1e-8)  # Normalize

        for move, prior in zip(moves, priors):
            node.children[move] = Node(prior)
        
        node.expanded = True
        return value

    def _simulate(self, node, board):
        """Run a single MCTS simulation from the root and backpropagate the value."""
        node.N += 1
        if board.is_game_over() or board.is_repetition(3) or board.can_claim_fifty_moves():
            if board.is_checkmate():
                value = -1
            else:
                value = 0

            node.W += value
            node.Q = node.W / node.N
            return value

        if not node.expanded:
            value = self._expand(node, board)
            node.W += value
            node.Q = node.W / node.N
            return value

        move, child = self._select(node)
        board.push(move)
        value = -self._simulate(child, board)
        board.pop()

        node.W += value
        node.Q = node.W / node.N
        return value

    def _add_dirichlet_noise(self, node, alpha=0.3, epsilon=0.25):
        """Add Dirichlet noise to the root node's priors for exploration."""
        moves = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for move, n in zip(moves, noise):
            node.children[move].P = (
                (1 - epsilon) * node.children[move].P + epsilon * n
            )

    def search(self, board, sims, add_noise=True):
        """Perform MCTS simulations starting from the given board state."""
        root = Node(0.0)
        self._expand(root, board)
        if add_noise:
            self._add_dirichlet_noise(root)
        for _ in range(sims):
            self._simulate(root, board.copy())
        return {m: c.N for m, c in root.children.items()}


def get_best_move(choices, temperature):
    """After simulations, select the move with the highest visit count."""
    moves, counts = zip(*choices.items())
    counts = np.array(counts, dtype=np.float32)

    if temperature == 0:
        return moves[np.argmax(counts)]

    counts = counts ** (1 / (temperature))
    probs = counts / counts.sum()

    try:
        best_move = np.random.choice(moves, p=probs)
    except ValueError:
        print(counts, probs, choices)

    return best_move
