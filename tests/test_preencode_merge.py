import json
import pathlib
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.data.preencode import parse_merges

CSV = pathlib.Path("data/lichess_db_puzzle.csv")


def test_parse_merges():
    assert parse_merges(["winning=crushing,advantage"]) == {"crushing": "winning", "advantage": "winning"}
    assert parse_merges(None) == {}


@pytest.mark.skipif(not CSV.exists(), reason="raw CSV not present")
def test_preencode_merge_produces_winning(tmp_path):
    out, meta = tmp_path / "b.h5", tmp_path / "m.json"
    subprocess.run([sys.executable, "-m", "src.data.preencode", "--csv", str(CSV), "--limit", "3000", "--workers", "2",
                    "--out", str(out), "--meta", str(meta), "--merge", "winning=crushing,advantage"], check=True, capture_output=True)
    m = json.loads(meta.read_text())
    assert "winning" in m["labels"] and "crushing" not in m["labels"] and "advantage" not in m["labels"]
    assert m["merges"] == {"crushing": "winning", "advantage": "winning"}
    assert m["counts"]["winning"] == m["merged_source_counts"]["crushing"] + m["merged_source_counts"]["advantage"]
    with h5py.File(out) as f:
        labels = json.loads(f.attrs["labels"]); t = f["targets"][:]
        j = labels.index("winning"); k = labels.index("mate")
        assert t[:, j].sum() == m["counts"]["winning"] and not (t[:, j] & t[:, k]).any()  # exclusive with mate
