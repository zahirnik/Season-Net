import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from torch.optim import lr_scheduler

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import torch
import torch.nn.functional as F

def qm_loss(
    forecast,     # [B, T, H, W]
    reference,    # [B, T, H, W]
    window=21,    # temporal window size
    quantiles=torch.linspace(0, 1, 10)
):
    """
    Local quantile matching loss using a centered sliding window of 21 days.

    Args:
        forecast:   predicted values [B, T, H, W]
        reference:  ground truth     [B, T, H, W]
        window:     temporal window length (must be odd for centered padding)
        quantiles:  torch tensor of quantile levels (e.g., 0.0 to 1.0)

    Returns:
        Scalar loss that penalizes misalignment of local distributions.
    """
    assert window % 2 == 1, "Window size must be odd for centered sliding"

    B, T, H, W = forecast.shape
    pad = window // 2
    quantiles = quantiles.to(forecast.device)

    # Reflect pad on time dimension
    forecast_padded = F.pad(forecast, (0, 0, 0, 0, pad, pad), mode='reflect')   # [B, T+2*pad, H, W]
    reference_padded = F.pad(reference, (0, 0, 0, 0, pad, pad), mode='reflect')

    total_loss = 0.0
    for t in range(T):
        f_win = forecast_padded[:, t:t+window]  # [B, window, H, W]
        r_win = reference_padded[:, t:t+window]

        f_q = torch.quantile(f_win, quantiles, dim=1)  # [Q, B, H, W]
        r_q = torch.quantile(r_win, quantiles, dim=1)

        loss = torch.abs(f_q - r_q).mean()  # scalar
        total_loss += loss

    return total_loss / T


def train_model(model, train_loader, val_loader, epochs=30, lr=1e-4, device="cuda", seed=None, model_idx=None):
    if seed is not None:
        set_seed(seed)

    opt_choice = "adamw"  # Options: 'adam', 'adamw', 'rmsprop'

    if opt_choice == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=1e-3
        )

    elif opt_choice == "adamw":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-2,
            weight_decay=1e-2
        )

    elif opt_choice == "rmsprop":
        optimizer = torch.optim.RMSprop(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=1e-2,
        )

    else:
        raise ValueError(f"Unknown opt_choice: {opt_choice}")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, total_batches = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).squeeze(1)
            preds = model(x)
            loss = qm_loss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        train_losses.append(avg_loss)

        model.eval()
        total_val_loss, val_batches = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).squeeze(1)
                preds = model(x)
                loss = qm_loss(preds, y)
                total_val_loss += loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        elapsed = time.time() - start_time

        mem_used = torch.cuda.memory_allocated(device) / (1024 ** 2) if device == "cuda" else 0.0
        mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device == "cuda" else 0.0

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.2f} sec | GPU Mem: {mem_used:.1f} MB | Reserved: {mem_reserved:.1f} MB", flush=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({opt_choice.upper()})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = f"training_validation_loss_{opt_choice}.png" if model_idx is None else f"training_validation_loss_{opt_choice}_{model_idx}.png"
    plt.savefig(fname)
    plt.close()
    print(f"📉 Saved {fname}")

    return model



def evaluate_model(model, test_loader, device="cuda"):
    model.eval()
    preds_full, targets_full = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).squeeze(1)
            preds = model(x)
            preds_full.append(preds.cpu())
            targets_full.append(y.cpu())
    preds_full = torch.cat(preds_full, dim=0)
    targets_full = torch.cat(targets_full, dim=0)
    return preds_full, targets_full
