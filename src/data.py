import numpy as np
import torch

from utils import load_config

CONFIG = load_config()

def load_raw_data(p_min: int, p_max: int, n_min: int, n_max: int, p_step: int, n_step: int) -> dict[int, np.ndarray]:
    raw_dir = CONFIG["paths"]["raw_dir"]
    data = {}
    for p in range(p_min, p_max + 1, p_step):
        file_dir = raw_dir / str(p)
        for n in range(n_min, n_max + 1, n_step):
            file_path = file_dir / f"{n}.csv"
            try:
                data[(p, n)] = np.loadtxt(file_path, delimiter=',')
            except Exception as e:
                print(f"Error loading data for N = {n} from {file_path}: {e}")
                data[(p, n)] = None

    return data

def get_n_nu(n: int) -> int:
    closest_magic = min(CONFIG["nuclei"]["magic_numbers"], key=lambda x: abs(n - x))
    return abs(n - closest_magic) // 2

def prepare_training_dataset(raw_dict: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ generate training dataset X, Y from raw_dict(N -> array[[beta, E(beta)], ...])
    
    X: (num_samples, 3) = [N, N_nu, beta]
    Y: (num_samples, 1) = E(beta)
    """
    X_rows: list[list[float]] = []
    Y_vals: list[float] = []
    for (p, n), arr in raw_dict.items():
        if arr is None:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[WARN] Skipped N={n} : expected 2 columns [beta,E], got shape {arr.shape}")
            continue
        n_nu = get_n_nu(int(n))
        beta_arr = arr[:, 0]
        HFB_energies = arr[:, 1]
        for beta, energy in zip(beta_arr, HFB_energies):
            X_rows.append([float(n), float(n_nu), float(beta)])
            Y_vals.append(float(energy))
    X = np.array(X_rows)
    Y = np.array(Y_vals)
    return X, Y

def save_processed_data(X: np.ndarray, Y: np.ndarray, basename: str) -> dict:
    """ save processed data X, Y to .npy and .csv files, return paths """
    processed_dir = CONFIG["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    x_path = processed_dir / f"{basename}_X.npy"
    y_path = processed_dir / f"{basename}_Y.npy"
    np.save(x_path, X)
    np.save(y_path, Y)
    paths["X"] = x_path
    paths["Y"] = y_path

    # for convenience of inspection / visualization, also save as CSV
    csv_path = processed_dir / f"{basename}.csv"
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "n_nu", "beta", "E"])  # header
            for (n, nn, b), e in zip(X, Y):
                writer.writerow([int(n), int(nn), float(b), float(e)])
        paths["csv"] = csv_path
    except Exception as e:
        print(f"[WARN] Failed to write CSV: {e}")
    return paths

def load_processed_data():
    """ load processed data X, Y from .npy files """
    processed_dir = CONFIG["paths"]["processed_dir"]
    x_path = processed_dir / "training_data_X.npy"
    y_path = processed_dir / "training_data_Y.npy"
    X = torch.from_numpy(np.load(x_path)).float()
    Y = torch.from_numpy(np.load(y_path)).float().unsqueeze(1)
    return X, Y

def _make_split_indices(X: torch.Tensor,
                        val_ratio: float,
                        seed: int) -> tuple:
    """ make train/val split indices """
    rng = np.random.default_rng(seed)

    neutrons = X[:, 0].detach().cpu().numpy().astype(int)
    unique_n = np.unique(neutrons)

    idx_train: list[int] = []
    idx_val: list[int] = []

    for n in unique_n:
        group_idx = np.where(neutrons == n)[0]
        group_size = group_idx.size
        if group_size == 0:
            continue
        rng.shuffle(group_idx)

        raw_val_count = int(round(group_size * val_ratio))
        val_count = max(0, min(group_size - 1, raw_val_count))

        if val_count > 0:
            idx_val.extend(group_idx[:val_count].tolist())
        idx_train.extend(group_idx[val_count:].tolist())

    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)
    return idx_train, idx_val

def main():
    raw_data = load_raw_data(
        CONFIG["nuclei"]["p_min"],
        CONFIG["nuclei"]["p_max"],
        CONFIG["nuclei"]["n_min"],
        CONFIG["nuclei"]["n_max"],
        CONFIG["nuclei"]["p_step"],
        CONFIG["nuclei"]["n_step"],
    )
    X, Y = prepare_training_dataset(raw_data)
    saved_paths = save_processed_data(X, Y, "training_data")
    print(f"Processed data saved: {saved_paths}")

if __name__ == "__main__":
    main()