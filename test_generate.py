# Helper script that does the generation of files for training whilst BeamNG is running.
import subprocess
import json
from pathlib import Path
import atexit

# Install zarr to make .zgroup marker file before being sent to the subprocess, use conda
try:
    import z5py
except ModuleNotFoundError:
    import sys, subprocess, importlib, shutil
    conda = shutil.which("conda")
    if not conda:
        raise SystemExit("conda not found on PATH, add package manually via \n conda install conda-forge::z5py")
    subprocess.check_call([conda, "install", "-y", "-c", "conda-forge", "z5py"])
    z5py = importlib.import_module("z5py")

# Call path for application
repo_root = Path(__file__).resolve().parent
application_path = (repo_root / "out/build/x64-debug/accdoa_gen.exe").resolve()
if not application_path.exists():
    raise FileNotFoundError(f"EXE not found: {application_path}")


# Iterative zarr path creation, keep ahold of path for zarr label operations
base_path = Path('trials').resolve()
base_path.mkdir(parents=True, exist_ok=True)

i = 1
while (base_path / f"trial_{i}.zarr").exists():
    i += 1

root_group = z5py.File(base_path / f"trial_{i}.zarr", mode='w')


# Intiation configuration sent to application
config_data = {
    "device_name": "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)",
    "zarr_path": str((base_path / f"trial_{i}.zarr").as_posix()),
}

process = subprocess.Popen(
    [str(application_path)],
    cwd=str(repo_root),
    stdin=subprocess.PIPE,
    stdout=None,   # inherit terminal (visible)
    stderr=None,   # inherit terminal (visible)
    text=True
)

def send_exit():
    if process.poll() is None and process.stdin:
        try:
            process.stdin.write("exit\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

atexit.register(send_exit)
process.stdin.write(json.dumps(config_data) + "\n")
process.stdin.flush()
process.wait()