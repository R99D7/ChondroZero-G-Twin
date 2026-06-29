import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictive_model import PredictiveRiskModel
from src.medical_agent import MedicalLLMAgent

st.set_page_config(layout="wide", page_title="ChondroZero-G Command Center", page_icon="🌌", initial_sidebar_state="expanded")

# --- Custom CSS (Glassmorphism & Deep Space Theme) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(14, 17, 30) 0%, rgb(0, 0, 0) 100%);
        color: #e0e6ed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(14, 17, 30, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics and Containers (Glassmorphism) */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00e5ff;
        text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.5);
    }
    .css-1r6slb0, .css-1y4p8pa { 
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Header Styling */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 1px;
    }
    .neon-text {
        color: #00e5ff;
        text-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff;
    }
    
    /* Anomaly Log Console */
    .console-log {
        background-color: #0d1117;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #30363d;
        height: 150px;
        overflow-y: scroll;
        box-shadow: inset 0 0 10px #000;
    }
    .console-alert { color: #ff3333; }
    .console-warn { color: #ffaa00; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    risk_model = PredictiveRiskModel()
    risk_model.train_synthetic() 
    cmo_agent = MedicalLLMAgent(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return risk_model, cmo_agent

risk_model, cmo_agent = load_models()

def plot_radar_chart(telemetry, matrix_integrity):
    categories = ['Bone Density', 'GCR Load', '15-PGDH', 'Inflammation', 'Matrix Integrity']
    
    # Normalize values for radar (0 to 1 scale roughly)
    values = [
        max(0, (telemetry['bone_density'] + 4) / 4.5), # -4 to 0.5 -> 0 to 1
        min(1, telemetry['radiation'] / 1000.0), 
        min(1, telemetry['pgdh'] / 10.0),
        min(1, telemetry.get('inflammation_score', 0.5)),
        matrix_integrity
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 229, 255, 0.3)',
        line=dict(color='#00e5ff'),
        name='Current Biomarker State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig

def main():
    st.markdown("<h1 style='text-align: center;'><span class='neon-text'>ChondroZero-G</span> Autonomous Digital Twin V2</h1>", unsafe_allow_html=True)
    
    # --- Sidebar ---
    st.sidebar.markdown("### 🎛️ Mission Controls")
    node = st.sidebar.selectbox("Federated Network Node", ["Node-1: Artemis V", "Node-2: Mars Transit A", "Node-3: Mars Transit B"])
    
    scenario = st.sidebar.selectbox("Clinical Scenario Injection", [
        "Nominal Mars Transit (Baseline)",
        "Solar Particle Event (SPE) - Radiation Spike",
        "Severe Microgravity Osteopenia",
        "Inhibitor Toxicity / Hepatic Stress",
        "Supply Chain Failure (No Drug Available)"
    ])
    
    timeline = st.sidebar.slider("Mission Timeline (Months)", 0, 24, 6)
    
    # Network Status UI
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 FL Network Sync")
    st.sidebar.markdown(f"**Target Node:** `{node}`")
    sync_status = "ONLINE 🟢" if "Failure" not in scenario else "LAGGING 🟡"
    st.sidebar.markdown(f"**Status:** {sync_status}")
    st.sidebar.markdown(f"**Latency:** `{np.random.randint(150, 450)} ms`")
    st.sidebar.markdown(f"**Global Weights Aggregated:** `{timeline * 42} epochs`")

    # --- Telemetry Simulation Logic ---
    np.random.seed(hash(node + scenario) % (2**32 - 1) + timeline)
    
    current_telemetry = {
        'heart_rate': np.random.uniform(60, 80),
        'bone_density': np.random.uniform(-1.5, 0.0),
        'radiation': np.random.uniform(50, 150) * (timeline / 6),
        'pgdh': np.random.uniform(2, 5),
        'inflammation_score': np.random.uniform(0.2, 0.5)
    }
    
    anomalies = []
    
    if "Solar Particle Event" in scenario:
        current_telemetry['radiation'] += np.random.uniform(500, 1000)
        current_telemetry['pgdh'] += np.random.uniform(5, 10)
        current_telemetry['inflammation_score'] = 0.95
        anomalies.append(f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-alert'>CRITICAL: SPE Detected. Radiation flux +850%.</span>")
    elif "Severe Microgravity" in scenario:
        current_telemetry['bone_density'] = np.random.uniform(-3.5, -2.5)
        current_telemetry['pgdh'] += np.random.uniform(3, 6)
        anomalies.append(f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-warn'>WARNING: Bone Density T-Score collapsed below -2.5 threshold.</span>")
    elif "Inhibitor Toxicity" in scenario:
        current_telemetry['heart_rate'] = np.random.uniform(90, 120) 
        current_telemetry['pgdh'] = np.random.uniform(0.1, 1.0) 
        anomalies.append(f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-alert'>CRITICAL: Hepatic stress biomarkers elevated. 15-PGDH dangerously suppressed.</span>")
    else:
        anomalies.append(f"[{datetime.now().strftime('%H:%M:%S')}] Nominal telemetry stream received.")
        
    rad_factor = current_telemetry['radiation'] / 1000.0
    bone_factor = abs(min(0, current_telemetry['bone_density'])) / 5.0
    matrix_integrity = max(0.01, 1.0 - rad_factor - bone_factor - (timeline/48))
    
    if "Supply Chain Failure" in scenario:
        recommended_drug_dose = 0.0
        matrix_integrity *= 0.5
        anomalies.append(f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-warn'>WARNING: Dispenser inventory query failed. Drug unavailable.</span>")
    elif "Inhibitor Toxicity" in scenario:
        recommended_drug_dose = 0.0
    else:
        recommended_drug_dose = min(50.0, current_telemetry['pgdh'] * 2.5 + (timeline))

    ode_trajectory = {
        "final_matrix_integrity": matrix_integrity,
        "recommended_drug_dose": recommended_drug_dose
    }
    
    risk_score = risk_model.predict_risk(current_telemetry)
    if "Supply Chain Failure" in scenario: risk_score = min(1.0, risk_score + 0.3)
    if "Inhibitor Toxicity" in scenario: risk_score = min(1.0, risk_score + 0.4)
    if "Solar Particle" in scenario: risk_score = min(1.0, risk_score + 0.5)

    # --- Top Row: Core Metrics & Radar ---
    col_metrics, col_radar, col_log = st.columns([1, 1.2, 1])
    
    with col_metrics:
        st.markdown("### 📊 Vital Telemetry")
        st.metric("XGBoost Mission Risk", f"{risk_score:.2f}", delta=f"{risk_score - 0.5:.2f}", delta_color="inverse")
        st.metric("Matrix Integrity Prediction", f"{matrix_integrity*100:.1f}%")
        st.metric("GCR Exposure (mSv)", f"{current_telemetry['radiation']:.1f}")
        
    with col_radar:
        st.markdown("### 🧬 Biomarker Profile (6-Axis)")
        fig = plot_radar_chart(current_telemetry, matrix_integrity)
        st.plotly_chart(fig, use_container_width=True)
        
    with col_log:
        st.markdown("### 📡 Deep Space Anomaly Log")
        log_html = "<div class='console-log'>" + "<br>".join(anomalies) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Recommended 15-PGDH Inhibitor", f"{recommended_drug_dose:.1f} mg")

    st.markdown("---")

    # --- Middle Row: CMO & Deep Inferences ---
    st.markdown("### 🤖 Autonomous Medical Agent Insights")
    col_cmo, col_pathways = st.columns([2, 1])
    
    with col_cmo:
        with st.spinner("Synthesizing clinical recommendation..."):
            summary = cmo_agent.generate_cmo_summary(current_telemetry, risk_score, ode_trajectory)
            
        if "ERROR" in summary or summary == "":
            if "Inhibitor Toxicity" in scenario:
                summary = f"**Simulated Summary:** ALERT - Severe hepatic stress detected. 15-PGDH is over-suppressed. HALT 15-PGDH inhibitor immediately despite {ode_trajectory['final_matrix_integrity']*100:.1f}% matrix integrity. Risk: {risk_score:.2f}."
            elif "Supply Chain Failure" in scenario:
                summary = f"**Simulated Summary:** WARNING - 15-PGDH inhibitor supply depleted. Matrix integrity projected to collapse to {ode_trajectory['final_matrix_integrity']*100:.1f}%. Initiate load-bearing exercise protocols and structural bracing. Risk: {risk_score:.2f}."
            elif "Solar Particle Event" in scenario:
                summary = f"**Simulated Summary:** CRITICAL - Massive SPE radiation dose ({current_telemetry['radiation']:.0f} mSv) detected. Accelerating cartilage breakdown. Administer maximum safe emergency dose of {ode_trajectory['recommended_drug_dose']:.1f} mg inhibitor. Risk: {risk_score:.2f}."
            else:
                summary = f"**Simulated Summary:** Nominal transit. Radiation: {current_telemetry['radiation']:.0f} mSv. Projected matrix integrity: {ode_trajectory['final_matrix_integrity']*100:.1f}%. Recommend maintenance dose of {ode_trajectory['recommended_drug_dose']:.1f} mg inhibitor. Risk: {risk_score:.2f}."
        
        st.info(summary)
        
    with col_pathways:
        st.markdown("**🔬 Molecular Pathway Inferences:**")
        if "Solar Particle" in scenario:
            st.markdown("- ⬆️ **IL-1β & TNF-α:** Massively upregulated due to radiation.")
            st.markdown("- ⬆️ **MMP-13:** Overexpressed, destroying Type II collagen.")
            st.markdown("- ⬇️ **SOX9:** Downregulated (Chondrocyte apoptosis).")
        elif "Inhibitor Toxicity" in scenario:
            st.markdown("- ⬇️ **15-PGDH:** Critical suppression.")
            st.markdown("- ⬆️ **ALT/AST (Simulated):** Elevated hepatic enzymes.")
        else:
            st.markdown("- ↔️ **IL-6:** Stable base levels.")
            st.markdown("- ↔️ **Aggrecan:** Normal synthesis rate.")
            st.markdown("- ⬇️ **MMP-13:** Successfully inhibited by current dosing.")

    # --- Bottom Row: Neural ODE & Vision Hub ---
    st.markdown("---")
    st.markdown("### 📈 Neural ODE Trajectory & Predictive Vision")
    col_chart, col_vision = st.columns([1, 1])
    
    with col_chart:
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
        
        st.line_chart(df, height=350)
        
    with col_vision:
        col3, col4, col5 = st.columns(3)
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        x = np.linspace(0, 255, 256)
        y = np.linspace(0, 255, 256)
        X, Y = np.meshgrid(x, y)
        dummy_heatmap = np.stack([X, Y, 255-X], axis=-1).astype(np.uint8)
        
        # Color shift heatmap based on integrity
        if matrix_integrity < 0.4:
            dummy_heatmap = np.stack([np.full_like(X, 255), Y, X], axis=-1).astype(np.uint8) # High red
            
        with col3:
            st.markdown("<div style='font-size:0.8rem; text-align:center;'>Current SEM</div>", unsafe_allow_html=True)
            st.image(dummy_img, use_column_width=True)
            
        with col4:
            st.markdown("<div style='font-size:0.8rem; text-align:center;'>Score-CAM</div>", unsafe_allow_html=True)
            st.image(dummy_heatmap, use_column_width=True)
            
        with col5:
            st.markdown("<div style='font-size:0.8rem; text-align:center;'>Future Prediction</div>", unsafe_allow_html=True)
            future_img = (dummy_img * matrix_integrity).astype(np.uint8)
            st.image(future_img, use_column_width=True)
        
if __name__ == "__main__":
    main()
