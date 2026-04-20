# Helper script on how to interact with the trainer/interface
import subprocess
import json
from pathlib import Path
import atexit

# Call path for application
repo_root = Path(__file__).resolve().parent
application_path = (repo_root / "out/build/x64-debug/ACCDOA-libtorch.exe").resolve()
if not application_path.exists():
    raise FileNotFoundError(f"EXE not found: {application_path}")


# Iterative zarr path creation, keep ahold of path for zarr label operations
base_path = Path('trials').resolve()
base_path.mkdir(parents=True, exist_ok=True)
highest_num = 0
for child in base_path.iterdir():
    if child.is_dir() and child.name.startswith("trial_"):
        try:
            num = int(child.name.split("_")[1])
            highest_num = max(highest_num, num)
        except ValueError:
            pass
path = base_path / f"zarr_{highest_num + 1}"
path.mkdir(parents=True, exist_ok=True)

# Intiation configuration sent to application
config_data = {
    "device_name": "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)",
    "zarr_path": str(path.as_posix()),
    "training_mode": True
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