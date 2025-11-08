import yaml
import numpy as np

from utils import load_config

SETTINGS = {
    "N_MIN": 86,
    "N_MAX": 96,
    "N_STEP": 2,
}
CONFIG = load_config()

def load_raw_data(p_min: int, p_max: int, n_min: int, n_max: int, p_step: int,n_step: int) -> dict[int, np.ndarray]:
    raw_dir = CONFIG["paths"]["raw_dir"]
    data = {}
    for p in range(p_min, p_max + 1, p_step):
        for n in range(n_min, n_max + 1, n_step):
            file_path = raw_dir / str(p) / f"{n}.csv"
            try:
                data[n] = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
            except Exception as e:
                print(f"Error loading data for n={n} from {file_path}: {e}")
                data[n] = None
    
    return data

def compute_n_nu(n: int) -> int:
    closest_magic = min(CONFIG["nuclei"]["magic_numbers"], key=lambda x: abs(n - x))
    return abs(n - closest_magic) // 2

def prepare_training_dataset(raw_dict: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ generate training dataset X, Y from raw_dict(N -> array[[beta, E(beta)], ...])
    
    X: (num_samples, 3) = [N, N_nu, beta]
    Y: (num_samples, 1) = E(beta)
    """
    X_rows: list[list[float]] = []
    Y_vals: list[float] = []
    for N, arr in (raw_dict.items()):



def build_training_arrays(raw_dict: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[list[float]] = []
    Y_vals: list[float] = []
    for N, arr in sorted(raw_dict.items(), key=lambda kv: kv[0]):
        if arr is None:
            continue
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[WARN] Skipped N={N}: expected 2 columns [beta,E], got shape {arr.shape}")
            continue
        n_nu = compute_n_nu(int(N))
        beta = arr[:, 0]
        energy = arr[:, 1]
        for b, e in zip(beta, energy):
            X_rows.append([float(N), float(n_nu), float(b)])
            Y_vals.append(float(e))
    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_vals, dtype=np.float32)
    return X, Y

def save_processed_data(X: np.ndarray, Y: np.ndarray, basename: str = "training") -> dict:
    """processed ディレクトリに X/Y を保存（.npy と .csv）。"""
    processed_dir = CONFIG["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    x_path = processed_dir / f"{basename}_X.npy"
    y_path = processed_dir / f"{basename}_Y.npy"
    np.save(x_path, X)
    np.save(y_path, Y)
    paths["X"] = x_path
    paths["Y"] = y_path

    # 検査・可視化向けにCSVも出力
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

if __name__ == "__main__":
    # 1) 生データ読み込み
    raw = load_raw_data(SETTINGS["N_MIN"], SETTINGS["N_MAX"], SETTINGS["N_STEP"])
    # 2) 学習用X, Yを構築
    X, Y = build_training_arrays(raw)
    print(f"Built X shape={X.shape}, Y shape={Y.shape}")
    # 3) processed 保存
    out_paths = save_processed_data(X, Y, basename="training")
    print("Saved:", {k: str(v) for k, v in out_paths.items()})