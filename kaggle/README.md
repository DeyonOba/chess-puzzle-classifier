# Kaggle full-data training

1. `make data` locally (18 base planes by default), then upload the HDF5 + metadata + splits as the dataset `gideonoba/chess-puzzle-data-v2`
   (`kaggle datasets create -p data/kaggle_dataset`; the metadata file is written by the Makefile target
   `kaggle-dataset`). The `.h5` is 1.3 GB.
2. `kaggle kernels push -p kaggle` pushes and runs `kaggle_train.py` on a GPU (settings in
   `kernel-metadata.json`: GPU + internet, dataset attached). The run uses `experiment=kaggle`
   (see `configs/experiment/kaggle.yaml`): 11 h max, early stopping, resumable.
3. To continue in the next session, add the previous kernel's output as an input source
   (`"kernel_sources": ["gideonoba/chess-puzzle-classifier-full-data-training"]`) and push again; the script resumes
   from the newest `last.ckpt` it finds under `/kaggle/input`.
4. Download `logs/train/runs/q2lg_full_v2x` (or `_v2` for the base-plane file) from the output; evaluate locally with
   `python -m src.eval ckpt_path=.../best.ckpt` and `python -m src.analyze`.

Monitor: `kaggle kernels status gideonoba/chess-puzzle-classifier-full-data-training`; stop from the Kaggle UI.

The script prefers `puzzle_boards_v2.h5` (18 planes; `PLANES=extended` prefers the 26-plane `puzzle_boards_v2x.h5`), falls back to whichever file is attached, and sets `model.net.in_planes` to match. Dataset version 1 holds the base file, version 2 the v2x file; both full runs are in the README. Attach the version you want from the kernel's data panel, or re-upload with `make kaggle-dataset`.
