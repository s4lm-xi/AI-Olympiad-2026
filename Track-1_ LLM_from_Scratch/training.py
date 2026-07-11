import math
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataset, epochs=20, learning_rate=1e-3, batch_size=32, device=None, log_every=50):
    device = device or get_device()
    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - perplexity: {math.exp(avg_loss):.2f}")

    return history


@torch.no_grad()
def evaluate(model, dataset, batch_size=32, device=None):
    device = device or get_device()
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(loader))
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}
