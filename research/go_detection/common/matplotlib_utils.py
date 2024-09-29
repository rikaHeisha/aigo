import matplotlib
import numpy as np

def intensity_to_rgb(intensity: np.ndarray, color_map: str = "plasma") -> np.ndarray:
    if not np.all( (intensity >= 0.0) & (intensity <= 1.0) ):
        raise ValueError("Intensity needs to be between 0 and 1")
    
    color_map = matplotlib.colormaps[color_map]
    return color_map(intensity)[:, :3] # discard alpha value and only return rgb