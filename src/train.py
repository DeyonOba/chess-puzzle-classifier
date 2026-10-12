"""Train (and test) the puzzle theme classifier with Hydra + Lightning.

    python -m src.train                                   # defaults: q2l_grouped gw=0.5, weighted BCE
    python -m src.train experiment=bounded_laptop run_name=q2lg_s0
    python -m src.train experiment=full_gpu seed=1 run_name=q2lg_full_s1
    python -m src.train debug=default                     # 5-batch smoke run on the dev subset
    python -m src.train -m model.group_weight=0.25,0.5,1  # sweep

Outputs under logs/<task_name>/runs/<run_name>/: config.yaml (resolved),
csv metrics, checkpoints/{best,last}.ckpt, test_results.json.
"""

from __future__ import annotations

import json
import pathlib

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers  # noqa: E402


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    out_dir = pathlib.Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=instantiate_callbacks(cfg.get("callbacks")), logger=instantiate_loggers(cfg.get("logger"))
    )

    if cfg.get("train"):
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    score = None
    if cfg.get("test"):
        ckpt = "best" if cfg.get("train") and trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path else cfg.get("ckpt_path")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt)
        if model.test_results is not None:
            (out_dir / "test_results.json").write_text(json.dumps(model.test_results, indent=2))
            score = model.test_results["val_tuned"]["macro_f1"]
            print(f"test: tuned macro_f1={score:.4f} micro_f1={model.test_results['val_tuned']['micro_f1']:.4f} -> {out_dir}")
    return score


if __name__ == "__main__":
    main()
