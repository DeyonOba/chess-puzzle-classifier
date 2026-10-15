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
RUN_NAME = os.environ.get("RUN_NAME")  # default: q2lg_full_<v2|v2x>, set below once the data file is known
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

print("/kaggle/input tree:", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count("/") - 2
    if depth <= 2:
        print("  " * depth + os.path.basename(root) + "/", *[f for f in files][:8], flush=True)
PLANES = os.environ.get("PLANES", "base")  # base (default, 18 planes) | extended (26); falls back to whichever file is attached
preferred = "puzzle_boards_v2x.h5" if PLANES == "extended" else "puzzle_boards_v2.h5"
other = "puzzle_boards_v2.h5" if preferred.endswith("v2x.h5") else "puzzle_boards_v2x.h5"
h5s = glob.glob(f"/kaggle/input/**/{preferred}", recursive=True)
if not h5s:
    h5s = glob.glob(f"/kaggle/input/**/{other}", recursive=True)
    print(f"WARNING: {preferred} not attached, using {other}", flush=True)
assert h5s, "attach the chess-puzzle-data-v2 dataset (no puzzle_boards_v2.h5 / puzzle_boards_v2x.h5 under /kaggle/input)"
data_dir = os.path.dirname(h5s[0])
h5_name = os.path.basename(h5s[0])
suffix = "v2x" if h5_name.endswith("v2x.h5") else "v2"
planes = 26 if suffix == "v2x" else 18
RUN_NAME = RUN_NAME or f"q2lg_full_{suffix}"
data_overrides = [f"data.h5_path={h5s[0]}", f"data.splits_path={data_dir}/splits_v2.npz",
                  f"model.labels_path={data_dir}/puzzle_metadata_{suffix}.json", f"model.net.in_planes={planes}"]
print("planes:", planes, "| file:", h5_name, flush=True)
ckpts = sorted(glob.glob("/kaggle/input/**/last.ckpt", recursive=True), key=os.path.getmtime)
resume = [f"ckpt_path={ckpts[-1]}"] if ckpts else []
print("data:", data_dir, "| resume:", resume or "fresh run", flush=True)

os.chdir(CODE)
sh(sys.executable, "-m", "src.train", "experiment=kaggle", f"run_name={RUN_NAME}",
   f"paths.data_dir={data_dir}", f"paths.log_dir={WORK}/logs", *data_overrides, *resume, *sys.argv[1:])

# keep the output small: the run directory is what matters
shutil.rmtree(f"{CODE}/.git", ignore_errors=True)
print("done; outputs in", f"{WORK}/logs/train/runs/{RUN_NAME}", flush=True)
