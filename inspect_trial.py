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
    available = list(f.keys())
    if "sed_features" in f and "doa_features" in f:
        sed_f = f["sed_features"]
        doa_f = f["doa_features"]
        n = min(frame_count, int(sed_f.shape[1]))
        if n == 0:
            raise RuntimeError("No frames found to plot.")

        sed_slice = np.asarray(sed_f[0, :n, :], dtype=np.float32)
        doa_slice = np.asarray(doa_f[:, :n, :], dtype=np.float32)

        sed_fig, sed_ax = plt.subplots(1, 1, figsize=(7, 5))
        sed_im = sed_ax.imshow(sed_slice.T, aspect="auto", origin="lower", interpolation="nearest")
        sed_ax.set_title(f"SED features ({n} frames)")
        sed_ax.set_xlabel("Frame")
        sed_ax.set_ylabel("Mel bin")
        sed_fig.colorbar(sed_im, ax=sed_ax)

        sed_fig.tight_layout()
        sed_out_png = trial_path.parent / f"{trial_path.stem}_sed_features_{n}.png"
        sed_fig.savefig(sed_out_png, dpi=140)
        print(f"Saved plot: {sed_out_png}")
        plt.show()

        doa_channels = int(doa_slice.shape[0])
        doa_fig, doa_ax = plt.subplots(1, doa_channels, figsize=(5 * doa_channels, 5), squeeze=False)
        for ch in range(doa_channels):
            im = doa_ax[0, ch].imshow(
                doa_slice[ch].T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
            )
            doa_ax[0, ch].set_title(f"DOA ch{ch} ({n} frames)")
            doa_ax[0, ch].set_xlabel("Frame")
            doa_ax[0, ch].set_ylabel("Feature bin")
            doa_fig.colorbar(im, ax=doa_ax[0, ch])

        doa_fig.tight_layout()
        doa_out_png = trial_path.parent / f"{trial_path.stem}_doa_features_{n}.png"
        doa_fig.savefig(doa_out_png, dpi=140)
        print(f"Saved plot: {doa_out_png}")
        plt.show()

        mean_fig, mean_ax = plt.subplots(figsize=(10, 4))
        mean_ax.plot(np.mean(sed_slice, axis=1), label="sed mean", alpha=0.8, linewidth=0.6)
        for ch in range(doa_channels):
            mean_ax.plot(
                np.mean(doa_slice[ch], axis=1),
                label=f"doa ch{ch} mean",
                alpha=0.8,
                linewidth=0.6,
            )
        mean_ax.set_title(f"Per-frame means ({n} frames)")
        mean_ax.set_xlabel("Frame")
        mean_ax.set_ylabel("Mean value")
        mean_ax.legend()
        mean_fig.tight_layout()
        mean_out_png = trial_path.parent / f"{trial_path.stem}_features_mean_{n}.png"
        mean_fig.savefig(mean_out_png, dpi=140)
        print(f"Saved plot: {mean_out_png}")
        plt.show()
        return
    elif "sed_labels" in f and "doa_labels" in f:
        sed = f["sed_labels"]
        doa = f["doa_labels"]
        n = min(frame_count, int(sed.shape[1]))
        if n == 0:
            raise RuntimeError("No frames found to plot.")

        sed_slice = np.asarray(sed[0, :n, :], dtype=np.float32)
        doa_slice = np.asarray(doa[0, :n, :], dtype=np.float32)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        im0 = ax[0].imshow(sed_slice.T, aspect="auto", origin="lower", interpolation="nearest")
        ax[0].set_title(f"SED labels ({n} frames)")
        ax[0].set_xlabel("Frame")
        ax[0].set_ylabel("Class/track")
        fig.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(doa_slice.T, aspect="auto", origin="lower", interpolation="nearest")
        ax[1].set_title(f"DOA labels ({n} frames)")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Cartesian")
        fig.colorbar(im1, ax=ax[1])

        fig.tight_layout()
        out_png = trial_path.parent / f"{trial_path.stem}_labels_{n}.png"
        fig.savefig(out_png, dpi=140)
        print(f"Saved plot: {out_png}")
        plt.show()

        mean_fig, mean_ax = plt.subplots(figsize=(10, 4))
        mean_ax.plot(np.mean(sed_slice, axis=1), label="sed mean", alpha=0.8, linewidth=0.6)
        mean_ax.plot(np.mean(doa_slice, axis=1), label="doa mean", alpha=0.8, linewidth=0.6)
        mean_ax.set_title(f"Per-frame means ({n} frames)")
        mean_ax.set_xlabel("Frame")
        mean_ax.set_ylabel("Mean value")
        mean_ax.legend()
        mean_fig.tight_layout()
        mean_out_png = trial_path.parent / f"{trial_path.stem}_labels_mean_{n}.png"
        mean_fig.savefig(mean_out_png, dpi=140)
        print(f"Saved plot: {mean_out_png}")
        plt.show()
        return
    else:
        raise KeyError(f"No known datasets found. Available: {available}")



if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    trial_path = (project_root / 'trials').resolve()
    trial = Path(trial_path / "trial_1.zarr").resolve()
    plot_frames(trial, frame_count=10000)