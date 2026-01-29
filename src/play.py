"""Inference"""
import chess
import torch
import hyperparams as hp
from model import FastChessNet
from mcts import MCTS
from util import print_board, parse_move

# Load model
net = FastChessNet().to(hp.DEVICE)
net.load_state_dict(torch.load(hp.MODEL_PATH, map_location=hp.DEVICE))
net.eval()

# Initialize MCTS
mcts = MCTS(net, c_puct=hp.CPUCT)

def play():
    """Main game loop."""
    board = chess.Board()
    print("Enter moves in UCI format (e2e4, g1f3, etc).")
    print("You play White.\n")

    while not board.is_game_over():
        print_board(board)

        if board.turn == chess.WHITE:
            move = None
            while move is None:
                move = parse_move(board, input("Your move: ").strip())
            board.push(move)
        else:
            print("Engine thinking...")
            pi = mcts.search(board, hp.SIMULATIONS_INF)
            move = max(pi, key=pi.get)
            board.push(move)

    print_board(board)
    print("Game over:", board.result())

if __name__ == "__main__":
    play()
