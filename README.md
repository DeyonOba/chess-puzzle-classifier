```bash
chess-puzzle-classifier/
├── configs/
│   ├── datamodule/
│   │   └── default.yaml         # Dataset paths, batch sizes, worker counts
│   ├── model/
│   │   └── maia2_multilabel.yaml# Backbone channels, hidden dims, loss weights
│   └── config.yaml              # Global training runtime parameters
├── data/                        # Local raw CSV and generated HDF5 files (gitignored)
│   ├── lichess_db_puzzle.csv
│   ├── puzzle_boards.h5
│   └── puzzle_metadata.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preencode.py          # Offline CSV -> HDF5 conversion script
│   │   ├── dataset.py            # HDF5 PyTorch Dataset class
│   │   └── datamodule.py         # Multi-label stratified DataModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   └── resnet_board.py   # Board representation feature extractor
│   │   └── puzzle_classifier.py  # Model head (Sigmoid / ML-GCN / Transformer)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── bitboard_encoder.py   # Vectorized bitwise board-to-array translation
│   │   └── metrics.py            # Multi-label F1 (micro/macro) & subset accuracy
│   └── train.py                  # Entrypoint for training execution
├── .gitignore
├── pyproject.toml / requirements.txt
└── README.md
```