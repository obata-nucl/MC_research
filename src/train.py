import csv
import math
import multiprocessing as mp
import numpy as np
import random
import torch
import torch.optim as optim
import torch.multiprocessing as mp_torch

from data import load_training_dataset, minmax_scaler, apply_minmax_scaler, _make_split_indices
from losses import loss_fn
from model import NN
from torch.utils.data import TensorDataset, DataLoader, Subset
from utils import load_config, get_all_patterns, _pattern_to_name

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
    _set_seed(base_seed + process_id)
    print(f"Process {process_id} Training start. Pattern: {hidden_dims}")

    # Build a base dataset once and create lightweight Subsets for train/val
    base_dataset = TensorDataset(X, X_scaled, Y)
    training_dataset = Subset(base_dataset, idx_train.tolist())
    g = torch.Generator()
    g.manual_seed(base_seed + process_id)
    train_loader = DataLoader(
        training_dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=g,
    )
    val_dataset = Subset(base_dataset, idx_val.tolist())
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["training"]["batch_size"],
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

    loss_dir = CONFIG["paths"]["results_dir"] / "training" / _pattern_to_name(hidden_dims)
    loss_dir.mkdir(parents=True, exist_ok=True)
    loss_path = loss_dir / "loss.csv"

    best_val_loss = float("inf")
    no_improve = 0
    num_epochs = CONFIG["training"]["num_epochs"]
    early_patience = CONFIG["training"]["early_stopping_patience"]

    with open(loss_path, mode='w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_MSE", "val_MSE", "train_RMSE", "val_RMSE", "lr"])
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
                writer_csv.writerow([
                    epoch + 1,
                    f"{train_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{math.sqrt(max(train_loss, 0.0)):.8f}",
                    f"{math.sqrt(max(val_loss, 0.0)):.8f}",
                    f"{current_lr:.6g}",
                ])
                f.flush()

                # Early stopping
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

def run_training():
    X, Y = load_training_dataset()
    idx_train, idx_val = _make_split_indices(X, val_ratio=CONFIG["training"]["val_ratio"], seed=CONFIG["training"]["base_seed"])

    x_min, x_range = minmax_scaler(X[idx_train])
    X_scaled = apply_minmax_scaler(X, x_min, x_range)
    try:
        X.share_memory_()
        X_scaled.share_memory_()
        Y.share_memory_()
        idx_train.share_memory_()
        idx_val.share_memory_()
    except Exception:
        pass

    patterns = get_all_patterns(CONFIG["nn"]["nodes_options"], CONFIG["nn"]["layers_options"])
    num_patterns = len(patterns)
    try:
        torch.save({"min": x_min, "range": x_range}, CONFIG["paths"]["results_dir"] / "scaler.pt")
    except Exception as e:
        print(f"Error saving scaler: {e}")
    
    print(f"Total patterns to train: {num_patterns}")

    args_lst = [
        (CONFIG["nn"]["input_dim"], pattern, CONFIG["nn"]["output_dim"],
         X, X_scaled, Y, idx_train, idx_val, id, CONFIG["training"]["base_seed"])
         for id, pattern in enumerate(patterns)
    ]

    num_cpus = mp.cpu_count()
    num_processes = max(1, num_cpus // 2)
    print(f"Starting training with {num_processes} processes")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_processes) as pool:
        pool.map(train_worker, args_lst)
    
    print("Training completed for all patterns")

def main():
    _set_seed(CONFIG["training"]["base_seed"])
    torch.set_num_threads(1)          # 行列・畳み込みなどのintra-op並列
    torch.set_num_interop_threads(1)  # operator間の並列
    try:
        mp_torch.set_sharing_strategy('file_system')
    except Exception:
        pass

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    run_training()

if __name__ == "__main__":
    main()