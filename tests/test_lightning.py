import pathlib

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV_H5 = ROOT / "data/dev/puzzle_boards_100k.h5"


def _cfg(overrides):
    import os
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=overrides, return_hydra_config=False)


@pytest.mark.parametrize("net", ["q2l_grouped", "q2l", "flat", "grouped", "mlgcn"])
def test_config_composes_and_instantiates_model(net, tmp_path):
    cfg = _cfg([f"model/net={net}", "model.net.hidden_dim=16", "model.net.num_res_blocks=1", "model.net.max_positions=4",
                "model.labels_path=null", f"paths.output_dir={tmp_path}"])
    cfg.model.labels = ["mate", "crushing", "advantage", "fork", "pin", "mateIn1", "mateIn2"]
    model = instantiate(cfg.model)
    assert model.net.head_type == net and model.hparams.net_spec["kwargs"]["head"] == net
    raw = model(torch.rand(2, 4, 18, 8, 8).round(), torch.tensor([1, 4]))
    assert model.probs(raw).shape == (2, 7)


@pytest.mark.skipif(not DEV_H5.exists(), reason="dev HDF5 not built")
def test_debug_experiment_fast_run(tmp_path):
    import lightning as L
    cfg = _cfg(["debug=default", f"paths.output_dir={tmp_path}", "~callbacks", "~logger"])
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, limit_train_batches=3, limit_val_batches=2, limit_test_batches=2,
                        logger=False, enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0,
                        use_distributed_sampler=False, default_root_dir=tmp_path)
    trainer.fit(model, datamodule=datamodule)
    # data-derived weights were filled from the training split
    assert model.criterion.free_loss.pos_weight.max() > 1.0 and model.criterion.class_weights[0].max() > 1.0
    assert (model.thresholds != 0.5).any()  # tuned on val
    trainer.test(model, datamodule=datamodule)
    r = model.test_results
    assert set(r) >= {"fixed_0.5", "val_tuned", "argmax_groups", "per_class", "ranking"}
    assert 0 <= r["argmax_groups"]["macro_f1"] <= 1
    # checkpoint round trip rebuilds the net from the stored recipe
    ckpt = tmp_path / "m.ckpt"; trainer.save_checkpoint(ckpt)
    from src.models.puzzle_module import PuzzleLitModule
    m2 = PuzzleLitModule.load_from_checkpoint(ckpt, map_location="cpu")
    assert m2.labels == model.labels and torch.equal(m2.thresholds, model.thresholds.cpu())
