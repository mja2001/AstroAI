import numpy as np
from astropy.modeling import models

def simulate_light_curve(period=10, radius_ratio=0.1, inclination=90, noise_level=0.01, num_points=1000):
    # Time array
    t = np.linspace(0, period * 2, num_points)
    
    # Simple transit model
    transit = models.Trapezoid1D(amplitude=-radius_ratio**2, duration=0.1*period, slope=1/period)
    phase = (t % period) / period - 0.5
    flux = 1 + transit(phase)  # Normalized flux
    
    # Add noise (Gaussian)
    flux += np.random.normal(0, noise_level, num_points)
    
    return t, flux
