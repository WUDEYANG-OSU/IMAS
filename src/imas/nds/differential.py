from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from imas.utils.stats import bh_fdr


def compute_group_deg(normal: np.ndarray, tumour: np.ndarray, genes, pseudocount: float = 0.02):
    normal=np.asarray(normal,float); tumour=np.asarray(tumour,float)
    mean_n,mean_t=normal.mean(0),tumour.mean(0); pct_n=(normal>0).mean(0); pct_t=(tumour>0).mean(0)
    _,p=stats.ttest_ind(tumour,normal,axis=0,equal_var=False,nan_policy="omit")
    return pd.DataFrame({"gene":list(map(str,genes)),"mean_normal":mean_n,"mean_tumor":mean_t,
                         "pct_normal":pct_n,"pct_tumor":pct_t,"log2FC_tumor_vs_normal":np.log2((mean_t+pseudocount)/(mean_n+pseudocount)),
                         "pvalue":p,"fdr":bh_fdr(np.nan_to_num(p,nan=1.0))})


def pseudobulk_welch(normal_samples: np.ndarray, tumour_samples: np.ndarray, genes, pseudocount: float = 0.5):
    normal=np.asarray(normal_samples,float); tumour=np.asarray(tumour_samples,float)
    _,p=stats.ttest_ind(np.log2(tumour+pseudocount),np.log2(normal+pseudocount),axis=0,equal_var=False,nan_policy="omit")
    logfc=np.log2((tumour.mean(0)+pseudocount)/(normal.mean(0)+pseudocount))
    return pd.DataFrame({"gene":list(map(str,genes)),"log2FC":logfc,"pvalue":p,"padj":bh_fdr(np.nan_to_num(p,nan=1.0))})
