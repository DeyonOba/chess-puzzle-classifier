# chess-puzzle-classifier
#   make setup     install deps (uv)
#   make data      download + decompress + encode HDF5 (v2 vocabulary) + splits   [skips what exists]
#   make dev       100k-row dev subset for smoke runs and tests
#   make test      pytest
#   make train     python -m src.train $(ARGS)
#   make serve     uvicorn with CKPT=... [POS_CKPT=...]
#   make clean-data / clean-logs

UV        ?= uv
PY         = $(UV) run python
DATA_DIR  ?= data
CSV        = $(DATA_DIR)/lichess_db_puzzle.csv
ZST        = $(CSV).zst
H5         = $(DATA_DIR)/puzzle_boards_v2.h5
META       = $(DATA_DIR)/puzzle_metadata_v2.json
SPLITS     = $(DATA_DIR)/splits_v2.npz
MERGE     ?= winning=crushing,advantage
T_MAX     ?= 12
DEV_N     ?= 100000
DEV_H5     = $(DATA_DIR)/dev/puzzle_boards_$(shell echo $(DEV_N) | sed 's/000$$/k/').h5
DEV_META   = $(DATA_DIR)/dev/puzzle_metadata_$(shell echo $(DEV_N) | sed 's/000$$/k/').json
DEV_SPLITS = $(DATA_DIR)/dev/splits_$(shell echo $(DEV_N) | sed 's/000$$/k/').npz
ARGS      ?=
PORT      ?= 8000

.PHONY: setup data download decompress encode splits dev test train eval serve clean-data clean-logs help

help:
	@sed -n '2,9p' Makefile | sed 's/^# *//'

setup:
	$(UV) sync

# ---- data --------------------------------------------------------------------------------------
data: $(SPLITS)

download: $(ZST)
$(ZST):
	scripts/fetch_puzzles.sh $(DATA_DIR)

decompress: $(CSV)
$(CSV): $(ZST)
	scripts/decompress_zstd.sh $(DATA_DIR) $(notdir $(ZST)) $(DATA_DIR)

encode: $(H5)
$(H5) $(META): $(CSV)
	$(PY) -m src.data.preencode --csv $(CSV) --out $(H5) --meta $(META) --max-positions $(T_MAX) --merge $(MERGE)

splits: $(SPLITS)
$(SPLITS): $(H5)
	$(PY) -m src.data.components.splits --h5 $(H5) --out $(SPLITS)

dev: $(DEV_SPLITS)
$(DEV_H5) $(DEV_META): $(CSV)
	mkdir -p $(DATA_DIR)/dev
	$(PY) -m src.data.preencode --csv $(CSV) --limit $(DEV_N) --out $(DEV_H5) --meta $(DEV_META) --max-positions $(T_MAX) --merge $(MERGE)
$(DEV_SPLITS): $(DEV_H5)
	$(PY) -m src.data.components.splits --h5 $(DEV_H5) --out $(DEV_SPLITS)

# ---- work --------------------------------------------------------------------------------------
test:
	$(UV) run pytest -q

train:
	$(PY) -m src.train $(ARGS)

eval:
	@test -n "$(CKPT)" || (echo "usage: make eval CKPT=logs/train/runs/NAME/checkpoints/best.ckpt"; exit 1)
	$(PY) -m src.eval ckpt_path=$(CKPT) $(ARGS)

serve:
	@test -n "$(CKPT)" || (echo "usage: make serve CKPT=... [POS_CKPT=...] [PORT=8000]"; exit 1)
	CPC_CKPT=$(CKPT) CPC_POSITION_CKPT=$(POS_CKPT) $(UV) run uvicorn src.app:app --port $(PORT)

clean-data:
	rm -f $(H5) $(META) $(SPLITS)
clean-logs:
	rm -rf logs
