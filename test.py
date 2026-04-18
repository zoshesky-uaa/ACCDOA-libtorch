# Helper script on how to interact with the trainer/interface
import subprocess
import json
from pathlib import Path

# Call path for application
application_path = Path('./ACCDOA-libtorch.exe').resolve()

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
    application_path, 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True  # Treats input/output as strings rather than bytes
)

json_payload = json.dumps(config_data)
stdout, stderr = process.communicate(input=json_payload)