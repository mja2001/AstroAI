import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulator import simulate_light_curve
from model import TransitDetector
import torch

st.title("AstroAI: Exoplanet Detection Simulator")

# User inputs
period = st.slider("Orbital Period (days)", 1, 30, 10)
radius_ratio = st.slider("Planet Radius Ratio", 0.01, 0.2, 0.1)
inclination = st.slider("Inclination (degrees)", 0, 90, 90)
noise_level = st.slider("Noise Level", 0.001, 0.05, 0.01)

if st.button("Simulate Light Curve"):
    t, flux = simulate_light_curve(period, radius_ratio, inclination, noise_level)
    fig, ax = plt.subplots()
    ax.plot(t, flux)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Simulated Exoplanet Transit Light Curve')
    st.pyplot(fig)

    # Save for README (optional)
    fig.savefig('assets/sample_light_curve.png')

# AI Detection
if st.button("Detect Transit"):
    t, flux = simulate_light_curve(period, radius_ratio, inclination, noise_level)
    model = TransitDetector()
    model.eval()
    input_data = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)  # Assuming 1D input
    with torch.no_grad():
        prob = model(input_data).item()
    st.write(f"Transit Probability: {prob:.2f}")

# Educational Section
st.header("Learn More")
st.markdown("""
### Kepler's Laws
Kepler's third law: \( P^2 \propto a^3 \)

### Transit Depth
\(\delta = \left(\frac{R_p}{R_s}\right)^2\)
""")
