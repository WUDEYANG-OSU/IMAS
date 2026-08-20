from __future__ import annotations

import numpy as np

from imas.benchmark import PerturbationResponse


def test_perturbation_response_roundtrip(tmp_path):
    response = PerturbationResponse(
        method="test",
        target_gene="EGFR",
        gene_ids=np.array(["EGFR", "MYC"]),
        cell_ids=np.array(["c1", "c2"]),
        baseline_state=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        perturbed_state=np.array([[0.0, 2.5], [0.0, 4.5]], dtype=np.float32),
    )
    path = response.save_npz(tmp_path / "response.npz")
    loaded = PerturbationResponse.load_npz(path)
    loaded.validate()
    assert loaded.method == "test"
    assert loaded.target_gene == "EGFR"
    assert np.allclose(loaded.response_delta, response.response_delta)
