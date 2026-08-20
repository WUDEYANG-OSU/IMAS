from __future__ import annotations
import math
import numpy as np
from sklearn.preprocessing import StandardScaler


def _standardize_block(matrix):
    x=np.asarray(matrix,np.float32)
    if x.ndim==1:x=x[:,None]
    x=StandardScaler().fit_transform(x).astype(np.float32); x=np.clip(x,-10,10); return x/math.sqrt(max(1,x.shape[1]))


def build_mltd_mapped_representation(native_pca: np.ndarray, stageb_embedding: np.ndarray,
                                     communication_features: np.ndarray, temporal_features: np.ndarray,
                                     weights=(1.0,1.0,0.70,0.70)):
    blocks=[_standardize_block(block)*weight for block,weight in zip([native_pca,stageb_embedding,communication_features,temporal_features],weights)]
    return np.concatenate(blocks,axis=1).astype(np.float32)
