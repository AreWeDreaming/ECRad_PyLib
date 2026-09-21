import numpy as np
from scipy.interpolate import interp1d
import aug_sfutils as sf

def get_nearest_index(eq: sf.EQU, tarr: float|np.ndarray) -> np.ndarray:
    """Find nearest time index for a given time.
    """

    tim_eq = np.array(eq.time)
    tarr = np.atleast_1d(tarr)
    idx = interp1d(tim_eq, np.arange(len(tim_eq)), kind='nearest', assume_sorted=True)(tarr)
    return np.array(idx, dtype=int)