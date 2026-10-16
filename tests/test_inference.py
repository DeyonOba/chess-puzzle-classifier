import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEV_H5 = ROOT / "data/dev/puzzle_boards_100k.h5"
FEN_CSV = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
SETUP, LINE = "f2g3", ["e6e7", "b2b1", "b3c1", "b1c1", "h6c1"]
MATE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"   # Ra8# in one


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    if not DEV_H5.exists():
        pytest.skip("dev HDF5 not built")
    import lightning as L
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import os
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    out = tmp_path_factory.mktemp("run")
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train.yaml", overrides=["debug=default", f"paths.output_dir={out}", "~callbacks", "~logger"])
    dm, model = instantiate(cfg.data), instantiate(cfg.model)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, limit_train_batches=2, limit_val_batches=2, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0, use_distributed_sampler=False)
    trainer.fit(model, datamodule=dm)
    path = out / "best.ckpt"; trainer.save_checkpoint(path)
    return path


def test_predictor_line_and_rules(ckpt):
    from src.inference import Predictor
    p = Predictor(ckpt, device="cpu")
    out = p.predict(FEN_CSV, LINE, setup_move=SETUP)
    assert out.line_provided and out.positions == 6 and out.model == "line"
    assert "long" in out.tags and "long" in out.rule_tags            # 5 plies -> long (cook.py counts the setup move)
    assert not any(t.startswith("mateIn") for t in out.tags)          # line does not end in mate
    assert len(out.probabilities) == len(p.labels) and all(0 <= v <= 1 for v in out.probabilities.values())
    # argmax decoding: at most one tag per exclusive group
    from src.utils.label_groups import EXCLUSIVE_GROUPS
    for members in EXCLUSIVE_GROUPS.values():
        assert sum(t in out.tags for t in members) <= 1


def test_predictor_cp_rule_and_mate_rule(ckpt):
    from src.inference import Predictor
    p = Predictor(ckpt, device="cpu")
    out = p.predict(FEN_CSV, LINE, cp=850, setup_move=SETUP)
    assert "crushing" in out.tags and "advantage" not in out.tags and out.rule_tags["crushing"].startswith("rule: cp")
    out = p.predict(FEN_CSV, LINE, cp=300, setup_move=SETUP)
    assert "advantage" in out.tags and "crushing" not in out.tags
    out = p.predict(MATE_FEN, ["a1a8"])
    assert {"mate", "mateIn1", "oneMove"} <= set(out.tags) and out.rule_tags["mateIn1"] == "rule: line ends in checkmate"


def test_predictor_fen_only_and_no_rules(ckpt):
    from src.inference import Predictor
    p = Predictor(ckpt, device="cpu")
    out = p.predict(MATE_FEN)
    assert not out.line_provided and out.positions == 1 and not out.rule_tags
    out2 = p.predict(MATE_FEN, ["a1a8"], apply_rules=False, decoding="independent")
    assert out2.decoding == "independent" and not out2.rule_tags


def test_fastapi_app(ckpt, monkeypatch):
    from fastapi.testclient import TestClient
    import src.app as appmod
    monkeypatch.setenv("CPC_CKPT", str(ckpt)); monkeypatch.setenv("CPC_DEVICE", "cpu")
    with TestClient(appmod.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.get("/labels").json()["labels"]) == 70
        r = client.post("/predict", json={"fen": FEN_CSV, "setup_move": SETUP, "moves": LINE, "cp": 850})
        assert r.status_code == 200 and "crushing" in r.json()["tags"] and r.json()["positions"] == 6
        assert client.post("/predict", json={"fen": "not a fen"}).status_code == 422
        assert client.post("/predict", json={"fen": MATE_FEN, "moves": ["a1a7", "zz"]}).status_code == 422


def test_predict_batch_matches_single_predictions_and_embeds(ckpt):
    from src.inference import PredictItem, Predictor
    p = Predictor(ckpt, device="cpu")
    items = [PredictItem(FEN_CSV, LINE, setup_move=SETUP), PredictItem(MATE_FEN, ["a1a8"]), PredictItem(MATE_FEN)]
    batch = p.predict_batch(items, include_embedding=True)
    assert [b.model for b in batch] == ["line", "line", "line"]        # no position checkpoint: the line model answers all
    assert [b.positions for b in batch] == [6, 2, 1]
    for item, b in zip(items, batch):
        single = p.predict(item.fen, item.moves, setup_move=item.setup_move)
        assert b.tags == single.tags and b.rule_tags == single.rule_tags
        assert all(abs(b.probabilities[l] - single.probabilities[l]) < 1e-4 for l in p.labels)
        assert len(b.embedding) == p.embedding_dim and all(isinstance(v, float) for v in b.embedding)
    assert p.predict(MATE_FEN).embedding is None
    assert batch[0].embedding != batch[1].embedding
    with pytest.raises(ValueError, match="decoding"):
        p.predict_batch(items, decoding="nope")


def test_forward_with_embedding_is_masked_mean_over_valid_positions(ckpt):
    from src.inference import PredictItem, Predictor
    p = Predictor(ckpt, device="cpu")
    one = p.predict_batch([PredictItem(MATE_FEN)], include_embedding=True)[0]
    two = p.predict_batch([PredictItem(MATE_FEN, ["a1a8"])], include_embedding=True)[0]
    raw, emb = p.model.net.forward_with_embedding(*p.encode(MATE_FEN, None)[:2])
    assert raw.shape == (1, len(p.labels) + p.layout.num_groups) and emb.shape == (1, p.embedding_dim)
    assert torch.allclose(emb[0], torch.tensor(one.embedding), atol=1e-4)
    assert one.embedding != two.embedding


def test_predict_item_check_reports_bad_input():
    from src.inference import PredictItem
    assert PredictItem(MATE_FEN, ["a1a8"]).check() is None
    assert "zz" in (PredictItem(MATE_FEN, ["a1a7", "zz"]).check() or "")
    assert PredictItem("not a fen").check() is not None


def test_fastapi_batch_endpoint(ckpt, monkeypatch):
    from fastapi.testclient import TestClient
    import src.app as appmod
    monkeypatch.setenv("CPC_CKPT", str(ckpt)); monkeypatch.setenv("CPC_DEVICE", "cpu")
    with TestClient(appmod.app) as client:
        health = client.get("/health").json()
        assert health["embedding_dim"] > 0 and health["checkpoint"] and health["max_batch"] == appmod.MAX_BATCH
        r = client.post("/predict_batch", json={"items": [
            {"fen": FEN_CSV, "setup_move": SETUP, "moves": LINE, "cp": 850},
            {"fen": MATE_FEN, "moves": ["a1a7", "zz"]},
            {"fen": MATE_FEN, "moves": ["a1a8"]},
        ], "include_embedding": True})
        assert r.status_code == 200
        body = r.json()
        assert body["checkpoint"] == health["checkpoint"] and len(body["results"]) == 3
        first, bad, mate = body["results"]
        assert "crushing" in first["prediction"]["tags"] and len(first["prediction"]["embedding"]) == body["embedding_dim"]
        assert bad["prediction"] is None and "zz" in bad["error"]
        assert "mateIn1" in mate["prediction"]["tags"]
        single = client.post("/predict", json={"fen": FEN_CSV, "setup_move": SETUP, "moves": LINE, "include_embedding": True}).json()
        assert len(single["embedding"]) == body["embedding_dim"]
        assert client.post("/predict_batch", json={"items": []}).status_code == 422


def test_position_model_with_different_plane_count(ckpt, tmp_path):
    """Regression: a 26-plane position checkpoint beside an 18-plane line checkpoint must be encoded with its own planes."""
    import functools, lightning as L, torch
    from src.models.puzzle_module import PuzzleLitModule
    from src.models.components.puzzle_classifier import PuzzleSequenceClassifier
    from src.inference import Predictor
    line = PuzzleLitModule.load_from_checkpoint(ckpt, map_location="cpu")
    net = functools.partial(PuzzleSequenceClassifier, in_planes=26, hidden_dim=16, num_res_blocks=1, max_positions=12,
                            head="q2l_grouped", head_kwargs={"num_layers": 1, "num_heads": 2})
    pos = PuzzleLitModule(net=net, labels=line.labels)
    pos.thresholds.fill_(0.3)
    pos_ckpt = tmp_path / "pos26.ckpt"
    torch.save({"state_dict": pos.state_dict(), "hyper_parameters": dict(pos.hparams), "epoch": 0, "global_step": 0,
                "pytorch-lightning_version": L.__version__}, pos_ckpt)
    p = Predictor(ckpt, position_ckpt_path=pos_ckpt, device="cpu")
    assert p.model.net.in_planes == 18 and p.position_model.net.in_planes == 26
    out = p.predict(MATE_FEN)                       # FEN only -> the 26-plane position model
    assert out.model == "position" and out.positions == 1 and all(abs(v - 0.3) < 1e-6 for v in out.thresholds.values())
    out = p.predict(MATE_FEN, ["a1a8"])             # with a line -> the 18-plane line model
    assert out.model == "line" and out.positions == 2
