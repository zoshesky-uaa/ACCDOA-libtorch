import atexit
import json
import os
import subprocess
from pathlib import Path

# Audio configuration, FFT parameters
AUDIO_INPUT_DEVICE_NAME = "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)"

GEN_BINARY_PATH = "out/install/x64-debug/Generate_Package/bin/accdoa_gen.exe" 
TRAIN_BINARY_PATH = "out/install/x64-debug/Train_Package/bin/accdoa_train.exe"

sample_rate = 16000 # Sample rate for audio capture (e.g., 16000 Hz)
fft_size = 512 # FFT size for the STFT
mel_bins = 128 # Number of Mel bands for the log-mel spectrogram
hop_length = 160 # Hop length
target_res = 0.2 # Target output resolution in second (i.e. 0.2s for 200ms resolution)
batch_size = 24 # Batch size for training
se_count = 3 # Maximum unique sound events for SED head
track_count = 3 # Maximum amount of overlapping events for DOAE head

project_root = Path(__file__).resolve().parent
trial_path = (project_root / 'trials').resolve()

# Calculated/Constant parameters:
epochs = 50 # Number of training epochs
warmup_epochs = 5 # Number of warmup epochs for learning rate scheduling
batch_amount = 5 # Number of batches to process for training;
channels = 4 # Number of audio channels (e.g., 4 for first-order ambisonics)
time_window = 3 # Time window for each training sample in seconds (e.g., 3 seconds)
patch_size = 16 # Patch size (P) (h x w kernel)
patch_overlap = 6 # Patch overlap (O) 
enc_layers = 12 # Encoder layers (L) 
# --------------------------------
# These are specific dimension for distilling from other transformer models:
att_headers = 12; # Attention heads (h) : 12 
embed_dim = 768 # (h x 64) 
# --------------------------------
input_frame_time = hop_length / sample_rate # Time per frame (Tpf)
frame_time_seq = int(time_window * (sample_rate / hop_length)) # Frames per time window 
frame_max = int(frame_time_seq * batch_size * batch_amount) # Simulation maximum length in frames, approximately 6 minutes
conv_stride = patch_size - patch_overlap #Convolution stride (S) : (P - O)
fft_bins = int(fft_size // 2 + 1) # Number of frequency bins from the FFT
history_size = int(fft_size - hop_length) # Number of samples that overlap between consecutive STFT frames

# Temporal (time-features) Patches (n_t) : 29 (floor((T - P) / S) + 1)
# Frequency (mel-features) Patches (n_f) : 12 (floor((M - P) / S) + 1))
# Total Patches (n) = (n_t * n_f)

n_t = int((time_window * (sample_rate / hop_length) - patch_size) / conv_stride + 1)
n_f = int((mel_bins - patch_size) / conv_stride + 1)
num_patches = int(n_t * n_f) # Total Patches (n) (n_t * n_f)
t_prime = int(time_window / target_res)
label_max = t_prime * batch_size * batch_amount
total_seq = t_prime + num_patches # Total sequence length (seq) (t' + n)
inference_amount = int(target_res * (sample_rate / hop_length)) # Number of frames to infer on per inference step (e.g., 10 for 100ms)

# SED Features (sed_featureset)
# Concept: 1-channel log-mel spectrogram.
#     read_buffer: [1, config.frame_time_seq, config.mel_bins] (e.g., [1, 300, 128])
# x_in: [config.batch_size, 1, config.frame_time_seq, config.mel_bins] (e.g., [24, 1, 300, 128])

# DOA Features (doa_featureset)
# Concept: 5-channel features (1 log-mel + 4 intensity vectors).
# read_buffer: [5, config.frame_time_seq, config.mel_bins] (e.g., [5, 300, 128])
# x_in: [config.batch_size, 5, config.frame_time_seq, config.mel_bins] (e.g., [24, 5, 300, 128])

# SED Labels (sed_labelset)
# Concept: Binary flag reference per class track label.
# read_buffer: [1, config.frame_time_seq, (se_count * track_count * 1)]
# x_in: [config.batch_size, 1, config.frame_time_seq, (se_count * track_count * 1)]

# DOA Labels (doa_labelset)
# Concept: Flattened Cartesian coordinates (X, Y).
# read_buffer: [1, config.frame_time_seq, (se_count * track_count * 2)]
# x_in: [config.batch_size, 1, config.frame_time_seq, (se_count * track_count * 2)]

sed_fet_buffer_dim = (1, frame_time_seq, mel_bins) # SED feature buffer dimension
doa_fet_buffer_dim = (5, frame_time_seq, mel_bins) # DOA feature buffer dimension
sed_label_buffer_dim = (1, t_prime, int(se_count * track_count * 1)) # SED label buffer dimension
doa_label_buffer_dim = (1, t_prime, int(se_count * track_count * 2)) # DOA label buffer dimension

MAXIMUM_VEHICLES = track_count * se_count # Maximum total vehicles in the simulation, considering all classes


const_json = {    
    "config": {
        "sample_rate": sample_rate,
        "fft_size": fft_size,
        "mel_bins": mel_bins,
        "hop_length": hop_length,
        "target_res": target_res,
        "batch_size": batch_size,
        "se_count": se_count,
        "track_count": track_count,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "batch_amount": batch_amount,
        "channels": channels,
        "time_window": time_window,
        "patch_size": patch_size,
        "patch_overlap": patch_overlap,
        "enc_layers": enc_layers,
        "att_headers": att_headers,
        "embed_dim": embed_dim,
        "input_frame_time": input_frame_time,
        "frame_time_seq": frame_time_seq,
        "frame_max": frame_max,
        "conv_stride": conv_stride,
        "fft_bins": fft_bins,
        "history_size": history_size,
        "t_prime": t_prime,
        "label_max": label_max,
        "inference_amount": inference_amount,
        "n_t": n_t,
        "n_f": n_f,
        "num_patches": num_patches,
        "total_seq": total_seq,
        
        # Lists/Tuples natively convert to std::vector in nlohmann
        "sed_fet_buffer_dim": list(sed_fet_buffer_dim),
        "doa_fet_buffer_dim": list(doa_fet_buffer_dim),
        "sed_label_buffer_dim": list(sed_label_buffer_dim),
        "doa_label_buffer_dim": list(doa_label_buffer_dim)
    }
}

# z5py is required to create the Zarr store before the generator starts
try:
    import z5py
except ModuleNotFoundError:
    raise SystemExit(
        "z5py not found. Install it with: conda install -c conda-forge z5py"
    )


def create_zarr_store(zarr_path: Path, config: dict):
    """Create the Zarr store and required datasets for labels."""
    se_count = int(config["se_count"])
    track_count = int(config["track_count"])
    label_max = int(config["label_max"])
    sed_label_buffer_dim = tuple(config["sed_label_buffer_dim"])
    doa_label_buffer_dim = tuple(config["doa_label_buffer_dim"])

    root_group = z5py.File(zarr_path, use_zarr_format=True)
    root_group.create_dataset(
        "sed_labels",
        dtype="float32",
        shape=(sed_label_buffer_dim[0], label_max, sed_label_buffer_dim[2]),
        chunks=sed_label_buffer_dim,
        compression="blosc",
        codec="zstd",
        fillvalue=0.0,
        clevel=3,
        shuffle=1,
        blocksize=0,
    )
    root_group.create_dataset(
        "doa_labels",
        dtype="float32",
        shape=(doa_label_buffer_dim[0], label_max, doa_label_buffer_dim[2]),
        chunks=doa_label_buffer_dim,
        compression="blosc",
        codec="zstd",
        fillvalue=0.0,
        clevel=3,
        shuffle=1,
        blocksize=0,
    )
    return root_group


# Call path for application
repo_root = Path(__file__).resolve().parent
application_path = (repo_root / GEN_BINARY_PATH).resolve()
if not application_path.exists():
    raise FileNotFoundError(f"EXE not found: {application_path}")


config = const_json["config"]

# Iterative zarr path creation, keep ahold of path for zarr label operations
base_path = trial_path
base_path.mkdir(parents=True, exist_ok=True)

i = 1
while (base_path / f"trial_{i}.zarr").exists():
    i += 1

zarr_path = base_path / f"trial_{i}.zarr"
create_zarr_store(zarr_path, config)

# Initiation configuration sent to application
config_data = {
    "device_name": AUDIO_INPUT_DEVICE_NAME,
    "zarr_path": str(zarr_path.as_posix()),
    **config,
}

process = subprocess.Popen(
    [str(application_path)],
    cwd=str(repo_root),
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,  # Intercept stdout instead of skipping it
    stderr=None,  # Inherit terminal (visible)
    text=True,
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

# Read the process output using a standard iterator (yields lines as they appear)
started = False
for line in process.stdout:
    line = line.strip()
    if not line:
        continue
    match line:
        case "START":
            print("\n[Python] Intercepted START.")
            started = True
        case "END":
            print("\n[Python] Intercepted END. Sequence complete.")
            break 
        case line if started and line.isdigit():
            pass
        case _:
            continue