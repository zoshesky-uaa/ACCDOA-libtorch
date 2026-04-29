import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Built with AI to verify data from a trial file.
def _ensure_module(module_name: str, conda_pkg: str, pip_pkg: str | None = None):
    pip_pkg = pip_pkg or conda_pkg
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    conda = shutil.which("conda")
    if conda:
        try:
            subprocess.check_call([conda, "install", "-y", "-c", "conda-forge", conda_pkg])
            return importlib.import_module(module_name)
        except Exception:
            pass

    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_pkg])
    return importlib.import_module(module_name)


np = _ensure_module("numpy", "numpy")
z5py = _ensure_module("z5py", "z5py")
plt = _ensure_module("matplotlib.pyplot", "matplotlib", "matplotlib")


def plot_frames(trial_path: Path, frame_count = 500) -> None:
    f = z5py.File(str(trial_path), mode="r")
    features = f["features"]

    n = min(frame_count, int(features.shape[1]))
    if n == 0:
        raise RuntimeError("No frames found to plot.")

    mel_100 = np.asarray(features[0, :n, :], dtype=np.float32)
    ivx_100 = np.asarray(features[1, :n, :], dtype=np.float32)
    ivy_100 = np.asarray(features[2, :n, :], dtype=np.float32)

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    im0 = ax[0, 0].imshow(mel_100.T, aspect="auto", origin="lower", interpolation="nearest")
    ax[0, 0].set_title(f"MEL ({frame_count} frames)")
    ax[0, 0].set_xlabel("Frame")
    ax[0, 0].set_ylabel("Mel bin")
    fig.colorbar(im0, ax=ax[0, 0])

    im1 = ax[0, 1].imshow(ivx_100.T, aspect="auto", origin="lower", interpolation="nearest")
    ax[0, 1].set_title(f"IV_X ({frame_count} frames)")
    ax[0, 1].set_xlabel("Frame")
    ax[0, 1].set_ylabel("FFT bin")
    fig.colorbar(im1, ax=ax[0, 1])

    im2 = ax[1, 0].imshow(ivy_100.T, aspect="auto", origin="lower", interpolation="nearest")
    ax[1, 0].set_title(f"IV_Y ({frame_count} frames)")
    ax[1, 0].set_xlabel("Frame")
    ax[1, 0].set_ylabel("FFT bin")
    fig.colorbar(im2, ax=ax[1, 0])

    ax[1, 1].plot(np.mean(mel_100, axis=1), label="mel mean", alpha=0.8, linewidth=0.4)
    ax[1, 1].plot(np.mean(ivx_100, axis=1), label="iv_x mean", alpha=0.8, linewidth=0.4)
    ax[1, 1].plot(np.mean(ivy_100, axis=1), label="iv_y mean", alpha=0.8, linewidth=0.4)
    ax[1, 1].set_title(f"Per-frame means ({frame_count} frames)")
    ax[1, 1].set_xlabel("Frame")
    ax[1, 1].set_ylabel("Mean value")
    ax[1, 1].legend()

    fig.tight_layout()
    out_png = trial_path.parent / f"{trial_path.stem}_{frame_count}.png"
    fig.savefig(out_png, dpi=140)
    print(f"Saved plot: {out_png}")
    plt.show()


if __name__ == "__main__":
    trial = Path("trials/trial_1.zarr").resolve()
    plot_frames(trial, frame_count=10000)