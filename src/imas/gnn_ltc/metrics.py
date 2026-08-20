from __future__ import annotations
import numpy as np


def _smooth(values, k=3):
    values=np.asarray(values,float)
    if k<=1:return values
    kernel=np.ones(k)/k
    return np.convolve(np.pad(values,(k//2,k//2),mode="edge"),kernel,mode="valid")[:len(values)]


def stage_metrics(time: np.ndarray, values: np.ndarray, coverage_threshold: float = 0.5):
    order=np.argsort(time); t=np.asarray(time)[order]; y=_smooth(np.asarray(values)[order])
    baseline=float(np.quantile(y,0.1)); high=float(np.quantile(y,0.9)); amplitude=high-baseline
    norm=(y-baseline)/(amplitude+1e-12); coverage=float((norm>=coverage_threshold).mean())
    speed=np.gradient(y,t+np.arange(len(t))*1e-12); max_speed=float(np.max(np.abs(speed)))
    t50=float(t[np.argmin(np.abs(norm-0.5))])
    return {"amplitude":float(amplitude),"coverage":coverage,"max_speed":max_speed,"t50":t50}
