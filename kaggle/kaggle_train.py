"""Kaggle script kernel: full-data training of chess-puzzle-classifier (experiment=kaggle).

Attach the dataset <user>/chess-puzzle-data-v2 (puzzle_boards_v2.h5, puzzle_metadata_v2.json,
splits_v2.npz). Internet must be enabled (git clone + pip). To resume a previous session, add the
previous run's output (the kernel's own output or a dataset containing logs/.../last.ckpt) as an
input; the script finds last.ckpt under /kaggle/input automatically.
"""
import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/DeyonOba/chess-puzzle-classifier.git"
RUN_NAME = os.environ.get("RUN_NAME", "q2lg_full")
WORK = "/kaggle/working"
CODE = f"{WORK}/repo"


def sh(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


if os.path.exists(CODE):
    shutil.rmtree(CODE)
sh("git", "clone", "--depth", "1", REPO, CODE)
# only what Kaggle does not ship; torch/pandas/numpy/h5py come with the image
sh(sys.executable, "-m", "pip", "install", "-q", "lightning>=2.4", "hydra-core>=1.3", "omegaconf", "rootutils", "rich",
   "chess>=1.10", "torchmetrics>=1.4", "h5py", "tqdm")

data_dirs = [d for d in glob.glob("/kaggle/input/*") if os.path.exists(f"{d}/puzzle_boards_v2.h5")]
assert data_dirs, "attach the chess-puzzle-data-v2 dataset"
data_dir = data_dirs[0]
ckpts = sorted(glob.glob("/kaggle/input/**/last.ckpt", recursive=True), key=os.path.getmtime)
resume = [f"ckpt_path={ckpts[-1]}"] if ckpts else []
print("data:", data_dir, "| resume:", resume or "fresh run", flush=True)

os.chdir(CODE)
sh(sys.executable, "-m", "src.train", "experiment=kaggle", f"run_name={RUN_NAME}",
   f"paths.data_dir={data_dir}", f"paths.log_dir={WORK}/logs", *resume, *sys.argv[1:])

# keep the output small: the run directory is what matters
shutil.rmtree(f"{CODE}/.git", ignore_errors=True)
print("done; outputs in", f"{WORK}/logs/train/runs/{RUN_NAME}", flush=True)
