"""Kaggle script kernel: full-data POSITION-ONLY model (data.max_positions=1) for FEN-only requests.

Wraps kaggle/kaggle_train.py from the public repo with overrides. T=1 epochs are ~4x cheaper than
line epochs, so a 7 h session covers ~20 epochs; early stopping (patience 4 validations) usually ends it.
Uses whichever data file is attached (PLANES=extended prefers the v2x file; the dataset's current
version holds only that file). Resume: attach this kernel's output and push again.
"""
import os
import urllib.request

os.environ["RUN_NAME"] = "q2lg_full_pos"
os.environ["PLANES"] = "extended"
os.environ["CPC_OVERRIDES"] = "data.max_positions=1 trainer.max_epochs=25 trainer.max_time=00:07:00:00"

URL = "https://raw.githubusercontent.com/DeyonOba/chess-puzzle-classifier/main/kaggle/kaggle_train.py"
src = urllib.request.urlopen(URL).read().decode()
exec(compile(src, "kaggle_train.py", "exec"))
