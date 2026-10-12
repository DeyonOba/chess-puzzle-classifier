# chess-puzzle-classifier
#   make setup     install deps (uv)
#   make data      download + decompress + encode HDF5 (69 labels, attack-map planes) + splits   [skips what exists; PLANES=base for 18 planes]
#   make dev       100k-row dev subset for smoke runs and tests
#   make test      pytest
#   make train     python -m src.train $(ARGS)
#   make serve     uvicorn with CKPT=... [POS_CKPT=...]
#   make kaggle-dataset / kaggle-push / kaggle-status   (see kaggle/README.md)
#   make clean-data / clean-logs

UV        ?= uv
PY         = $(UV) run python
DATA_DIR  ?= data
CSV        = $(DATA_DIR)/lichess_db_puzzle.csv
ZST        = $(CSV).zst
PLANES    ?= extended            # extended = 18 base + 8 attack-map planes (default); base = 18
SUFFIX     = $(if $(filter extended,$(PLANES)),v2x,v2)
H5         = $(DATA_DIR)/puzzle_boards_$(SUFFIX).h5
META       = $(DATA_DIR)/puzzle_metadata_$(SUFFIX).json
SPLITS     = $(DATA_DIR)/splits_v2.npz
MERGE     ?= winning=crushing,advantage
T_MAX     ?= 12
DEV_N     ?= 100000
DEV_PLANES ?= base               # tests and debug=default expect the 18-plane dev file
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
	$(PY) -m src.data.preencode --csv $(CSV) --out $(H5) --meta $(META) --max-positions $(T_MAX) --merge $(MERGE) --planes $(PLANES)

splits: $(SPLITS)
$(SPLITS): $(H5)
	$(PY) -m src.data.components.splits --h5 $(H5) --out $(SPLITS)

dev: $(DEV_SPLITS)
$(DEV_H5) $(DEV_META): $(CSV)
	mkdir -p $(DATA_DIR)/dev
	$(PY) -m src.data.preencode --csv $(CSV) --limit $(DEV_N) --out $(DEV_H5) --meta $(DEV_META) --max-positions $(T_MAX) --merge $(MERGE) --planes $(DEV_PLANES)
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

# ---- kaggle -----------------------------------------------------------------------------------
KAGGLE     ?= kaggle
KAGGLE_USER ?= $(shell python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
STAGE       = $(DATA_DIR)/kaggle_dataset
.PHONY: kaggle-dataset kaggle-push kaggle-status
kaggle-dataset: $(SPLITS)           # stage v2 files (hard links) + metadata, then upload (~1.3 GB)
	mkdir -p $(STAGE)
	for f in $(notdir $(H5)) $(notdir $(META)) $(notdir $(SPLITS)); do ln -f $(DATA_DIR)/$$f $(STAGE)/$$f; done
	printf '{"title": "Lichess puzzle themes v2 (packed bitboards)", "id": "%s/chess-puzzle-data-v2", "licenses": [{"name": "CC0-1.0"}]}\n' $(KAGGLE_USER) > $(STAGE)/dataset-metadata.json
	$(KAGGLE) datasets create -p $(STAGE) --dir-mode zip || $(KAGGLE) datasets version -p $(STAGE) --dir-mode zip -m "update"
kaggle-push:                        # push + run the training kernel (uses the GPU budget)
	$(KAGGLE) kernels push -p kaggle
kaggle-status:
	$(KAGGLE) kernels status $(KAGGLE_USER)/chess-puzzle-classifier-full-data-training

