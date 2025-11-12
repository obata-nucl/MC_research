import numpy as np
import os
import signal
import subprocess
import torch

from data import load_eval_dataset
from losses import calc_sse
from model import load_NN_model
from utils import load_config

CONFIG = load_config()

def run_npbos(command: list[str], timeout_sec: float) -> tuple[str, str, int]:
    """ run NPBOS programs and return IBM spectra as stdout, stderr, and return code """
    proc = None
    try:
        proc = subprocess.Popen(
            list(map(str, command)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        rc = proc.returncode
        return (stdout or "", stderr or "", int(rc))
    except subprocess.TimeoutExpired:
        try:
            if proc is not None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            if proc is not None:
                proc.communicate(timeout=timeout_sec)
        except Exception:
            try:
                if proc is not None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        return ("", "timed out", -1)
    except Exception as e:
        try:
            if proc is not None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        return ("", str(e), -1)

def evaluate_model(X_eval: torch.Tensor, X_eval_scaled: torch.Tensor, pattern: list[int], expt_spectra: np.ndarray) -> tuple[float, float]:
    """ evaluate each pattern model """
    model = load_NN_model(pattern)
    if model is None:
        raise ValueError("Model could not be loaded.")
    model.eval()

    num_eval_X = len(X_eval)
    energy_RMSE, ratio_RMSE = 0.0, 0.0
    energy_count_total, ratio_count = 0, 0

    for i in range(num_eval_X):
        n = int(X_eval[i, 0].item())
        n_nu = int(X_eval[i, 1].item())
        expt_spectra_n = expt_spectra[expt_spectra[:, 0] == n]

        if expt_spectra_n.size == 0:
            print(f"No expt data found for N={n}")
            continue
        expt_energies = expt_spectra_n[0, 1:5]

        with torch.no_grad():
            outputs = model(X_eval_scaled[i].unsqueeze(0))
        
        pred_params = outputs.squeeze(0).numpy()
        sh_command = [
            "bash", "eval.sh",
            str(CONFIG["paths"]["NPBOS_dir"]),
            str(int(n + 62)), str(int(n_nu)),
            [f"`{param:.3f}" for param in pred_params]
        ]

        stdout, stderr, rc = run_npbos(sh_command)
        if rc != 0:
            print(f"timeout for N={n}, params = {pred_params}")
            continue
        try:
            pred_energies = list(map(float, stdout.strip().split()))
        except ValueError as e:
            print(f"Error parsing output: {stdout} - {e}")
            continue

        energy_sse, energy_count = calc_sse(pred_energies, expt_energies)
        energy_RMSE += energy_sse
        energy_count_total += energy_count
        # ratio evaluation
        if len(pred_energies) == len(expt_energies) and pred_energies[0] != 0 and expt_energies[0] != 0:
            ratio_RMSE += ((pred_energies[1] / pred_energies[0]) - (expt_energies[1] / expt_energies[0])) ** 2
            ratio_count += 1
    energy_RMSE = np.sqrt(energy_RMSE / energy_count_total) if energy_count_total > 0 else float('inf')
    ratio_RMSE = np.sqrt(ratio_RMSE / ratio_count) if ratio_count > 0 else float('inf')

    return energy_RMSE, ratio_RMSE