# Kaggle full-data training

1. `make data` locally, then upload the three v2 files as the dataset `gideonoba/chess-puzzle-data-v2`
   (`kaggle datasets create -p data/kaggle_dataset`; the metadata file is written by the Makefile target
   `kaggle-dataset`). The `.h5` is 1.3 GB.
2. `kaggle kernels push -p kaggle` pushes and runs `kaggle_train.py` on a GPU (settings in
   `kernel-metadata.json`: GPU + internet, dataset attached). The run uses `experiment=kaggle`
   (see `configs/experiment/kaggle.yaml`): 11 h max, early stopping, resumable.
3. To continue in the next session, add the previous kernel's output as an input source
   (`"kernel_sources": ["gideonoba/chess-puzzle-classifier-train"]`) and push again; the script resumes
   from the newest `last.ckpt` it finds under `/kaggle/input`.
4. Download `logs/train/runs/q2lg_full` from the output; evaluate locally with
   `python -m src.eval ckpt_path=.../best.ckpt` and `python -m src.analyze`.

Monitor: `kaggle kernels status gideonoba/chess-puzzle-classifier-train`; stop from the Kaggle UI.
