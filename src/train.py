import csv
import math
import multiprocessing as mp
import numpy as np
import random
import torch
import torch.optim as optim
import torch.multiprocessing as mp_torch

from data import _make_split_indices
from losses import loss_fn
from model import NN
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from utils import load_config, _pattern_to_name

CONFIG = load_config()

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _seed_worker(seed: int, worker_id: int):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_worker(args):
    input_dim, hidden_dims, output_dim, X, X_scaled, Y, idx_train, idx_val, process_id, base_seed = args
    print(f"Process {process_id} Training start. Pattern: {hidden_dims}")

    X_train, X_scaled_train, Y_train = X[idx_train], X_scaled[idx_train], Y[idx_train]
    X_val, X_scaled_val, Y_val = X[idx_val], X_scaled[idx_val], Y[idx_val]

    training_dataset = TensorDataset(X_train, X_scaled_train, Y_train)
    g = torch.Generator()
    g.manual_seed(base_seed + process_id)
    train_loader = DataLoader(
        training_dataset,
        batch_size=int(CONFIG["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, X_scaled_val, Y_val),
        batch_size=int(CONFIG["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = NN(input_dim, hidden_dims, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["training"]["lr_factor"],
        patience=CONFIG["training"]["lr_patience"],
        threshold=1e-4,
        cooldown=0,
        min_lr=1e-6,
    )
    try:
        pattern_name = _pattern_to_name(hidden_dims)
    except:
        pattern_name = str(hidden_dims)
    loss_dir = CONFIG["paths"]["results_dir"] / "training" / pattern_name
    # create run directory (per pattern)
    loss_dir.mkdir(parents=True, exist_ok=True)
    loss_path = loss_dir / "loss.csv"
    with open(loss_path, mode='w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_MSE", "val_MSE", "train_RMSE", "val_RMSE", "lr"])
    
    best_val_loss = float("inf")
    no_improve = 0
    num_epochs = int(CONFIG["training"]["num_epochs"])
    early_patience = int(CONFIG["training"]["early_stopping_patience"])
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss_sum = 0.0
            num_train_samples = 0
            for batch_X, batch_X_scaled, batch_Y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X_scaled)
                train_loss = loss_fn(outputs, 6, batch_X[:, 1], batch_X[:, 2], batch_Y)
                train_loss.backward()
                optimizer.step()
                bs = batch_X.size(0)
                train_loss_sum += train_loss.item() * bs
                num_train_samples += bs

            train_loss = train_loss_sum / max(1, num_train_samples)

            model.eval()
            val_loss_sum = 0.0
            num_val_samples = 0
            with torch.no_grad():
                for batch_X, batch_X_scaled, batch_Y in val_loader:
                    outputs = model(batch_X_scaled)
                    val_loss = loss_fn(outputs, 6, batch_X[:, 1], batch_X[:, 2], batch_Y)
                    bs = batch_X.size(0)
                    val_loss_sum += val_loss.item() * bs
                    num_val_samples += bs
            val_loss = val_loss_sum / max(1, num_val_samples)

            scheduler.step(val_loss)

            # Logging
            current_lr = optimizer.param_groups[0]["lr"]
            with open(loss_path, mode='a', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([epoch + 1, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{math.sqrt(train_loss):.8f}", f"{math.sqrt(val_loss):.8f}", f"{current_lr:.6g}"])

            # Early stopping (best checkpoint)
            if val_loss + 1e-12 < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), loss_dir / "best_model.pth")
            else:
                no_improve += 1
                if no_improve >= early_patience:
                    print(f"Process {process_id} Early stopping at epoch {epoch + 1}")
                    break
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
