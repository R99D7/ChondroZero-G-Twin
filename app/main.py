import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictive_model import PredictiveRiskModel
from src.medical_agent import MedicalLLMAgent

# For a production dashboard, we'd initialize models here and use st.cache_resource
# However, to avoid long load times on boot, we simulate some heavy initializations.

st.set_page_config(layout="wide", page_title="ChondroZero-G Digital Twin", page_icon="🚀")

@st.cache_resource
def load_models():
    risk_model = PredictiveRiskModel()
    risk_model.train_synthetic() # Train immediately on load
    # Note: MedicalLLMAgent initialization might be slow due to downloading weights.
    # We will instantiate it, but fallback to simulated text if it fails to load quickly.
    cmo_agent = MedicalLLMAgent(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return risk_model, cmo_agent

risk_model, cmo_agent = load_models()

def main():
    st.title("🚀 ChondroZero-G: Autonomous Digital Twin")
    st.markdown("### Decentralized Clinical Decision Support System for Deep Space Joint Health")

    # --- Sidebar ---
    st.sidebar.header("Mission Controls")
    node = st.sidebar.selectbox("Spacecraft Node Selector (Federated Network)", ["Node-1: Artemis V", "Node-2: Mars Transit A", "Node-3: Mars Transit B"])
    
    scenario = st.sidebar.selectbox("Simulated Clinical Scenario", [
        "Nominal Mars Transit (Baseline)",
        "Solar Particle Event (SPE) - Radiation Spike",
        "Severe Microgravity Osteopenia",
        "Inhibitor Toxicity / Hepatic Stress",
        "Supply Chain Failure (No Drug Available)"
    ])
    
    timeline = st.sidebar.slider("Mission Timeline (Months)", 0, 24, 6)
    
    # Simulate Telemetry based on Scenario and Timeline
    np.random.seed(hash(node + scenario) % (2**32 - 1) + timeline)
    
    # Default baseline
    current_telemetry = {
        'heart_rate': np.random.uniform(60, 80),
        'bone_density': np.random.uniform(-1.5, 0.0),
        'radiation': np.random.uniform(50, 150) * (timeline / 6),
        'pgdh': np.random.uniform(2, 5)
    }
    
    if "Solar Particle Event" in scenario:
        current_telemetry['radiation'] += np.random.uniform(500, 1000)
        current_telemetry['pgdh'] += np.random.uniform(5, 10)
    elif "Severe Microgravity" in scenario:
        current_telemetry['bone_density'] = np.random.uniform(-3.5, -2.5)
        current_telemetry['pgdh'] += np.random.uniform(3, 6)
    elif "Inhibitor Toxicity" in scenario:
        current_telemetry['heart_rate'] = np.random.uniform(90, 120) # Stress
        current_telemetry['pgdh'] = np.random.uniform(0.1, 1.0) # Over-suppressed
    
    # Calculate ODE outcomes
    rad_factor = current_telemetry['radiation'] / 1000.0
    bone_factor = abs(min(0, current_telemetry['bone_density'])) / 5.0
    
    matrix_integrity = max(0.01, 1.0 - rad_factor - bone_factor - (timeline/48))
    
    if "Supply Chain Failure" in scenario:
        recommended_drug_dose = 0.0
        matrix_integrity *= 0.5 # rapid degradation without drug
    elif "Inhibitor Toxicity" in scenario:
        recommended_drug_dose = 0.0 # Stop drug
    else:
        recommended_drug_dose = min(50.0, current_telemetry['pgdh'] * 2.5 + (timeline))

    ode_trajectory = {
        "final_matrix_integrity": matrix_integrity,
        "recommended_drug_dose": recommended_drug_dose
    }

    # --- Top Row: CMO Summary & Risk Gauges ---
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("👨‍⚕️ Autonomous CMO Executive Summary")
        risk_score = risk_model.predict_risk(current_telemetry)
        
        # Adjust risk score based on extreme cases
        if "Supply Chain Failure" in scenario: risk_score = min(1.0, risk_score + 0.3)
        if "Inhibitor Toxicity" in scenario: risk_score = min(1.0, risk_score + 0.4)
        if "Solar Particle" in scenario: risk_score = min(1.0, risk_score + 0.5)
        
        with st.spinner("Synthesizing clinical recommendation..."):
            summary = cmo_agent.generate_cmo_summary(current_telemetry, risk_score, ode_trajectory)
            
        if "ERROR" in summary or summary == "":
            if "Inhibitor Toxicity" in scenario:
                summary = f"**Simulated Summary:** ALERT - Severe hepatic stress detected (Heart Rate: {current_telemetry['heart_rate']:.0f} bpm). 15-PGDH is over-suppressed. HALT 15-PGDH inhibitor immediately despite {ode_trajectory['final_matrix_integrity']*100:.1f}% matrix integrity. Risk: {risk_score:.2f}."
            elif "Supply Chain Failure" in scenario:
                summary = f"**Simulated Summary:** WARNING - 15-PGDH inhibitor supply depleted. Matrix integrity projected to collapse to {ode_trajectory['final_matrix_integrity']*100:.1f}%. Initiate load-bearing exercise protocols and structural bracing. Risk: {risk_score:.2f}."
            elif "Solar Particle Event" in scenario:
                summary = f"**Simulated Summary:** CRITICAL - Massive SPE radiation dose ({current_telemetry['radiation']:.0f} mSv) detected. Accelerating cartilage breakdown. Administer maximum safe emergency dose of {ode_trajectory['recommended_drug_dose']:.1f} mg inhibitor. Risk: {risk_score:.2f}."
            else:
                summary = f"**Simulated Summary:** Nominal transit. Radiation: {current_telemetry['radiation']:.0f} mSv. Projected matrix integrity: {ode_trajectory['final_matrix_integrity']*100:.1f}%. Recommend maintenance dose of {ode_trajectory['recommended_drug_dose']:.1f} mg inhibitor. Risk: {risk_score:.2f}."
        
        st.info(summary)
        
    with col2:
        st.subheader("⚠️ Mission-Failure Risk")
        st.metric(label="XGBoost Risk Score", value=f"{risk_score:.2f} / 1.0", delta=f"{risk_score - 0.5:.2f}", delta_color="inverse")
        st.progress(risk_score)

    # --- Middle Row: Neural ODE Trajectory Charts ---
    st.markdown("---")
    st.subheader("📈 Neural ODE Trajectory (Cartilage vs. 15-PGDH vs. Drug)")
    
    # Simulate time series data for the charts
    t = np.linspace(0, timeline, 100)
    matrix_degrad = np.exp(-0.05 * t * (current_telemetry['radiation']/200))
    pgdh_levels = 1 - np.exp(-0.1 * t) + (current_telemetry['radiation']/1000)
    drug_levels = np.sin(t) * 0.5 + 0.5
    
    df = pd.DataFrame({
        'Time (Months)': t,
        'Matrix Integrity': matrix_degrad,
        '15-PGDH Concentration': pgdh_levels,
        'Drug Counter-measure': drug_levels
    }).set_index('Time (Months)')
    
    st.line_chart(df)

    # --- Bottom Row: The Vision Hub ---
    st.markdown("---")
    st.subheader("🔬 The Vision Hub: Predictive SEM Generative Modeling")
    
    col3, col4, col5 = st.columns(3)
    
    # Generate dummy images for visualization (in a real app, these are loaded/generated)
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # create a gradient for heatmap
    x = np.linspace(0, 255, 256)
    y = np.linspace(0, 255, 256)
    X, Y = np.meshgrid(x, y)
    dummy_heatmap = np.stack([X, Y, 255-X], axis=-1).astype(np.uint8)
    
    with col3:
        st.markdown("**Current SEM Upload**")
        st.image(dummy_img, caption="Baseline Tissue Matrix", use_column_width=True)
        
    with col4:
        st.markdown("**Score-CAM Degradation Heatmap**")
        st.image(dummy_heatmap, caption="High-Risk Areas Highlighted", use_column_width=True)
        
    with col5:
        st.markdown("**AI-Generated Future State (6 Months)**")
        # Darken dummy img to simulate degradation based on integrity
        future_img = (dummy_img * ode_trajectory['final_matrix_integrity']).astype(np.uint8)
        st.image(future_img, caption="Predicted Synthetic SEM (Diffusion Model)", use_column_width=True)
        
if __name__ == "__main__":
    main()
