"""Evaluate a checkpoint: re-tune per-class thresholds on the full val split, then score test.

    python -m src.eval ckpt_path=logs/train/runs/NAME/checkpoints/best.ckpt [data=...] [trainer=gpu]

Writes eval_results.json (metrics at 0.5, at val-tuned thresholds, and with
argmax-within-group decoding for grouped heads; per-class thresholds and F1)
into logs/eval/runs/<run_name>/.
"""

from __future__ import annotations

import json
import pathlib

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.puzzle_module import PuzzleLitModule  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    out_dir = pathlib.Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = PuzzleLitModule.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False, callbacks=[])

    trainer.validate(model, datamodule=datamodule)   # tunes thresholds on val (stored in model.thresholds)
    trainer.test(model, datamodule=datamodule)
    results = {"ckpt_path": str(cfg.ckpt_path), **model.test_results}
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    t = results["val_tuned"]; f = results["fixed_0.5"]
    print(f"test n={results['n_test']:,}\n  fixed 0.5 : micro_f1={f['micro_f1']:.4f} macro_f1={f['macro_f1']:.4f} exact={f['exact_match']:.4f}"
          f"\n  val-tuned : micro_f1={t['micro_f1']:.4f} macro_f1={t['macro_f1']:.4f} exact={t['exact_match']:.4f}")
    if "argmax_groups" in results:
        a = results["argmax_groups"]
        print(f"  argmax    : micro_f1={a['micro_f1']:.4f} macro_f1={a['macro_f1']:.4f} exact={a['exact_match']:.4f}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
