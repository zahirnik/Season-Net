import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import lr_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Directory Setup ===
main_directory = '/data//UK'
subdirectories = ['best_checkpoints', 'best_model', 'last_model', 'loss', 'Tensor_Out']
for sub in subdirectories:
    os.makedirs(os.path.join(main_directory, sub), exist_ok=True)

# === Loss Function: Quantile Mapping with Emphasis on Extremes ===
def quantile_mapping_loss_with_weights(forecast, reference, quantiles, land_mask, month_weights, sigma=0.2):
    B, T, H, W = forecast.shape
    quantiles_tensor = torch.tensor(quantiles, dtype=forecast.dtype, device=forecast.device)
    w_tau = 1 - 0.1 * torch.exp(-((quantiles_tensor.unsqueeze(-1).unsqueeze(-1) - 0.5) ** 2) / (2 * sigma ** 2))
    land_mask_expanded = land_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    time_steps = T // 3
    total_loss = 0.0
    for i in range(3):
        λ = month_weights[i]
        f_chunk = forecast[:, i * time_steps:(i + 1) * time_steps]
        r_chunk = reference[:, i * time_steps:(i + 1) * time_steps]
        f_q = torch.quantile(f_chunk, quantiles_tensor, dim=1)
        r_q = torch.quantile(r_chunk, quantiles_tensor, dim=1)
        error = torch.abs(f_q - r_q)
        weighted_error = w_tau * error
        weighted_error_land = weighted_error * land_mask_expanded
        total_loss += λ * weighted_error_land.mean()
    return total_loss

# === Utilities ===
def clip_gradient(model, clip_value):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(0.0001, clip_value)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class EarlyStopping:
    def __init__(self, tolerance=40, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
    def __call__(self, status):
        self.counter = 0 if status else self.counter + 1
        print(f"EarlyStopping counter: {self.counter}")
        if self.counter >= self.tolerance:
            self.early_stop = True

def save_loss_df(loss_stat, loss_df_name, loss_fig_name):
    df = pd.DataFrame.from_dict(loss_stat).reset_index().melt(id_vars=['index']).rename(columns={"index": "epochs"})
    df.to_csv(loss_df_name)
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="epochs", y="value", hue="variable").set_title('Train-Val Loss per Epoch')
    plt.ylim(0, df['value'].max())
    plt.savefig(loss_fig_name, dpi=300)

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_val_loss']

# === Bias Correction Model Wrapper ===
class BiasCorrect:
    def __init__(self, model, lr: float, wd: float, seeded: int, exp: str):
        self.model = model.apply(initialize_weights)
        self.lr = lr
        self.wd = wd
        self.exp = exp
        self.seeded = seeded
        self.optimizer = torch.optim.RMSprop(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.wd, momentum=0.7)
        self.step_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.01)
        self.best_model_name = os.path.join(main_directory, f'best_model_{exp}{seeded}.pth')
        self.last_model_name = os.path.join(main_directory, f'last_model_{exp}{seeded}.pth')
        self.best_checkpoint_dir = os.path.join(main_directory, f'best_checkpoints_{exp}{seeded}.pth')
        self.loss_fig_name = os.path.join(main_directory, 'loss', f'loss_{exp}{seeded}.png')
        self.loss_df_name = os.path.join(main_directory, 'loss', f'loss_{exp}{seeded}.csv')
        self.quantile_loss = quantile_mapping_loss_with_weights

    def train(self, train_loader, val_loader, reference_season, land_mask, epochs: int, loss_stop_tolerance: int):
        best_val_loss = float('inf')
        early_stopping = EarlyStopping(tolerance=loss_stop_tolerance)
        loss_stats = {'train': [], 'val': []}
        quantiles = list(np.linspace(0, 1, 21))
        month_weights = [0.3, 0.4, 0.3]

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.model.train()
            train_loss_epoch = 0
            train_inputs, train_targets = train_loader

            for inputs, targets in zip(train_inputs, train_targets):
                for x, y in zip(inputs, targets):
                    x = x.to(device).permute(1, 0, 4, 3, 2)
                    y = y.permute(0, 1, 3, 2).to(device)
                    preds = self.model(x)
                    loss = self.quantile_loss(preds, y, quantiles, land_mask, month_weights)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss_epoch += loss.item()

            self.step_lr_scheduler.step()

            # === Validation Phase ===
            self.model.eval()
            val_loss_epoch = 0
            val_inputs, val_targets = val_loader
            with torch.no_grad():
                for inputs, targets in zip(val_inputs, val_targets):
                    for x, y in zip(inputs, targets):
                        x = x.to(device).permute(1, 0, 4, 3, 2)
                        y = y.permute(0, 1, 3, 2).to(device)
                        preds = self.model(x)
                        loss = self.quantile_loss(preds, y, quantiles, land_mask, month_weights)
                        val_loss_epoch += loss.item()

            avg_train_loss = train_loss_epoch / len(train_inputs)
            avg_val_loss = val_loss_epoch / len(val_inputs)
            loss_stats['train'].append(avg_train_loss)
            loss_stats['val'].append(avg_val_loss)

            print(f"Epoch {epoch} | Time: {time.time() - start_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.best_model_name)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, filename=self.best_checkpoint_dir)
                print("Best model updated.")

            early_stopping(avg_val_loss < best_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        save_loss_df(loss_stats, self.loss_df_name, self.loss_fig_name)
        torch.save(self.model.state_dict(), self.last_model_name)

    def predict(self, model, data_loader_test, mini, maxi, seeded, Mode: str):
        """
        Generate predictions using best model. Save predictions to disk.
        """
        model.load_state_dict(torch.load(self.best_model_name))
        self.seeded = str(seeded)
        output_tensors = []

        # Normalize broadcast
        mini = mini.unsqueeze(0).repeat(4 if Mode == "train" else 1, 90, 1, 1).permute(0, 1, 3, 2)
        maxi = maxi.unsqueeze(0).repeat(4 if Mode == "train" else 1, 90, 1, 1).permute(0, 1, 3, 2)

        input_batches, target_batches = data_loader_test
        with torch.no_grad():
            for input_dl, target_dl in zip(input_batches, target_batches):
                for model_batch, _ in zip(input_dl, target_dl):
                    input_sequence = model_batch.to(device).permute(1, 0, 4, 3, 2)
                    output = model(input_sequence)
                    output = output * (maxi - mini) + mini  # unnormalize

                    if Mode == "train":
                        output_tensors.append(output)
                    else:
                        torch.save(output, os.path.join(main_directory, "Tensor_Out", f"{Mode}-{self.seeded}_Predict.pt"))

            if Mode == "train":
                full_output = torch.cat(output_tensors, dim=0)
                torch.save(full_output, os.path.join(main_directory, "Tensor_Out", f"{Mode}-{self.seeded}_Predict.pt"))
