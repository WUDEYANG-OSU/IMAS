import numpy as np
import pandas as pd
from scipy import sparse

from imas.data.gene_activity import build_peak_gene_matrix, compute_gene_activity
from imas.data.projection import project_sparse_csr_to_global
from imas.priors.reverse_edges import reverse_edge_table
from imas.priors.rna_re import build_rna_re_edges
from imas.priors.validation import validate_reverse_pair


def test_sparse_projection_merges_duplicate_targets():
    local = sparse.csr_matrix([[1.0, 2.0, 3.0]])
    projected, summary = project_sparse_csr_to_global(local, np.array([0, 0, 2]), 4)
    assert projected.shape == (1, 4)
    assert np.allclose(projected.toarray(), [[3.0, 0.0, 3.0, 0.0]])
    assert summary.duplicate_global_mappings == 1


def test_gene_activity_and_prior_composition():
    prior = build_peak_gene_matrix(
        np.array([0, 1]), np.array([0, 0]), np.array([1.0, 0.5]), n_peaks=2, n_genes=1
    )
    activity = compute_gene_activity(sparse.csr_matrix([[1.0, 2.0]]), prior)
    assert np.allclose(activity.toarray(), [[2.0]])

    rna_gene = pd.DataFrame({"src_name": ["RNA:D"], "dst_name": ["Gene:G"], "weight": [2.0]})
    re_gene = pd.DataFrame({"src_name": ["RE:P"], "dst_name": ["Gene:G"], "weight": [0.5]})
    rna_re = build_rna_re_edges(rna_gene, re_gene, normalize=False)
    assert rna_re.iloc[0]["weight"] == 1.0


def test_reverse_validation():
    forward = pd.DataFrame({"src_name": ["TF:A"], "dst_name": ["RE:B"], "weight": [0.7]})
    reverse = reverse_edge_table(forward)
    report = validate_reverse_pair(forward, reverse)
    assert report["matched_edges"] == 1
    assert report["weight_mismatches"] == 0
