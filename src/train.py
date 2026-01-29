"""Training scheme for FastChessNet using self-play."""
import torch
import torch.optim as optim

import hyperparams as hp
from model import FastChessNet
from selfplay import SelfPlayDataset


if hp.USE_SEED:
    torch.manual_seed(hp.SEED)


def train(model, replay_buffer):
    """Trains the network using data from the replay buffer."""
    optimizer = optim.Adam(model.parameters(), lr=hp.LR)
    criterion_policy = torch.nn.CrossEntropyLoss()
    criterion_value = torch.nn.MSELoss()

    for _ in range(hp.TRAINING_STEPS):
        state, mask, policy, value = replay_buffer.sample()
        state = torch.tensor(state).to(hp.DEVICE)
        mask = torch.tensor(mask).to(hp.DEVICE)
        policy = torch.tensor(policy).to(hp.DEVICE)
        value = torch.tensor(value).float().to(hp.DEVICE)

        out_p, out_v = model(state, mask)

        loss_v = criterion_value(out_v.squeeze(), value)
        loss_p = criterion_policy(out_p, policy.argmax(dim=1))
        loss = loss_v + loss_p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    """Main training loop."""
    model = FastChessNet().to(hp.DEVICE)
    model.load_state_dict(torch.load(hp.PT_MODEL_PATH, hp.DEVICE))
    model.train()
    buffer = SelfPlayDataset(model)

    print("Starting training...")

    for i in range(hp.ITERATIONS):
        for _ in range(hp.GAMES_PER_ITER):
            buffer.self_play()

        if len(buffer.buffer) >= hp.BATCH_SIZE:
            train(model, buffer)

        print("Iteration", i, "complete")
        if i % 10 == 0: # Save every 10 iterations
            torch.save(model.state_dict(), hp.MODEL_PATH)

    torch.save(model.state_dict(), hp.MODEL_PATH)

if __name__ == "__main__":
    main()
