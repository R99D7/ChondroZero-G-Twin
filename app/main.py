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

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(14, 17, 30) 0%, rgb(0, 0, 0) 100%);
        color: #e0e6ed;
    }
    [data-testid="stSidebar"] {
        background: rgba(14, 17, 30, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00e5ff;
        text-shadow: 0px 0px 10px rgba(0, 229, 255, 0.5);
    }
    .css-1r6slb0, .css-1y4p8pa, .st-emotion-cache-12w0qpk { 
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 1px;
    }
    .neon-text {
        color: #00e5ff;
        text-shadow: 0 0 5px #00e5ff, 0 0 10px #00e5ff;
    }
    .console-log {
        background-color: #0d1117;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #30363d;
        height: 250px;
        overflow-y: scroll;
        box-shadow: inset 0 0 10px #000;
        font-size: 0.85rem;
    }
    .console-alert { color: #ff3333; font-weight: bold; }
    .console-warn { color: #ffaa00; font-weight: bold; }
    .console-info { color: #00e5ff; }
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
    values = [
        max(0, (telemetry['bone_density'] + 4) / 4.5), 
        min(1, telemetry['radiation'] / 1000.0), 
        min(1, telemetry['pgdh'] / 10.0),
        min(1, telemetry.get('inflammation_score', 0.5)),
        matrix_integrity
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(0, 229, 255, 0.3)', line=dict(color='#00e5ff'),
        name='Current Biomarker State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), margin=dict(l=20, r=20, t=20, b=20), height=300
    )
    return fig

def plot_3d_surface():
    """Generates a 3D surface plot for deep inferences."""
    time_ax = np.linspace(0, 24, 30)
    rad_ax = np.linspace(0, 1000, 30)
    T, R = np.meshgrid(time_ax, rad_ax)
    Z = np.clip(1.0 - (R / 1000.0)**2 - (T / 48.0), 0, 1)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=T, y=R, colorscale='Viridis')])
    fig.update_layout(
        title='Cartilage Matrix Integrity Surface',
        scene=dict(
            xaxis_title='Time (Months)',
            yaxis_title='Radiation (mSv)',
            zaxis_title='Integrity (0-1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), margin=dict(l=0, r=0, b=0, t=30), height=400
    )
    return fig

def main():
    st.markdown("<h1 style='text-align: center;'><span class='neon-text'>ChondroZero-G</span> Autonomous Digital Twin V3</h1>", unsafe_allow_html=True)
    
    # --- Sidebar ---
    st.sidebar.markdown("### 🎛️ Mission Controls")
    node = st.sidebar.selectbox("Federated Network Node", [
        "Node-1: Artemis V", 
        "Node-2: Mars Transit A", 
        "Node-3: Mars Transit B",
        "Node-4: Orbital Reef Commercial Station",
        "Node-5: Titan Surface Base (Deep Cryo)",
        "Node-6: Europa Sub-Surface Submarine"
    ])
    
    scenario = st.sidebar.selectbox("Clinical Vulnerability Profile", [
        "Nominal Transit (Baseline)",
        "Solar Particle Event (SPE) - Radiation Spike",
        "Severe Microgravity Osteopenia",
        "Inhibitor Toxicity / Hepatic Stress",
        "Supply Chain Failure (No Drug Available)",
        "Cryogenic Joint Stiffening (Titan Methane Exposure)",
        "High-Pressure Decompression Stress (Europa)"
    ])
    
    st.sidebar.markdown("---")
    start_sim = st.sidebar.button("🚀 LAUNCH REAL-TIME SIMULATION", use_container_width=True, type="primary")
    
    st.sidebar.markdown("### 🌐 FL Network Sync")
    st.sidebar.markdown(f"**Target Node:** `{node}`")
    sync_status = "ONLINE 🟢" if "Failure" not in scenario else "LAGGING 🟡"
    
    # Titan/Europa nodes have massive latency
    latency_base = 150
    if "Titan" in node: latency_base = 4500000  # 75 minutes light travel time roughly
    elif "Europa" in node: latency_base = 2500000 # 40 minutes roughly
    st.sidebar.markdown(f"**Status:** {sync_status}")
    st.sidebar.markdown(f"**Latency:** `{latency_base + np.random.randint(-50, 50)} ms`")
    
    # --- Tabs ---
    tab_cmd, tab_mol, tab_vis, tab_fed = st.tabs([
        "🛰️ Command Center", "🧬 Deep Molecular Inference", "🔬 Generative Vision Lab", "⚙️ Federated Analytics"
    ])
    
    if not start_sim:
        with tab_cmd:
            st.info("System Standby. Click 'LAUNCH REAL-TIME SIMULATION' in the sidebar to begin dynamic telemetry streaming.")
        return

    # --- SIMULATION LOGIC ---
    with tab_cmd:
        col_metrics, col_radar, col_log = st.columns([1, 1.2, 1])
        metric_risk = col_metrics.empty()
        metric_integ = col_metrics.empty()
        metric_rad = col_metrics.empty()
        
        radar_ph = col_radar.empty()
        log_ph = col_log.empty()
        
        st.markdown("---")
        st.markdown("### 🤖 Autonomous Medical Agent (Live Updates)")
        cmo_ph = st.empty()
        
    with tab_mol:
        st.markdown("### 🔬 Biological Vulnerability Deep-Dive")
        vuln_desc = st.empty()
        st.plotly_chart(plot_3d_surface(), use_container_width=True)

    with tab_vis:
        st.markdown("### 🧠 Predictive Vision AI (Diffusion Outputs)")
        col_v1, col_v2, col_v3 = st.columns(3)
        v1_ph = col_v1.empty()
        v2_ph = col_v2.empty()
        v3_ph = col_v3.empty()
        
    with tab_fed:
        st.markdown("### 🌐 Decentralized AI Training Logs (Live Convergence)")
        st.markdown(f"**Target Node:** {node} | **Epoch Aggregation in Progress...**")
        loss_ph = st.empty()

    # Run Loop
    anomalies = []
    
    # Base telemetry for the simulation
    base_hr = 70
    base_bone = -0.5
    base_rad = 50
    base_pgdh = 3.0
    
    timeline_months = 0.0
    
    # Data structures for live line charts
    loss_data = []
    
    for i in range(1, 41): # 40 steps for smoother, longer simulation
        timeline_months += 0.5
        
        # Add noise for real-time feel
        hr = base_hr + np.random.uniform(-5, 5)
        bone = base_bone + np.random.uniform(-0.1, 0.1)
        rad = base_rad + (timeline_months * 5) + np.random.uniform(-10, 10)
        pgdh = base_pgdh + np.random.uniform(-0.5, 0.5)
        inf = 0.3 + np.random.uniform(-0.05, 0.05)
        
        # Apply Scenario Overrides
        if "Solar Particle" in scenario and timeline_months > 5:
            rad += np.random.uniform(300, 800)
            pgdh += 4.0
            inf = 0.9
            if i % 3 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-alert'>CRITICAL: SPE Flux Detected. GCR +{rad:.0f} mSv.</span>")
        elif "Severe Microgravity" in scenario:
            bone -= (timeline_months * 0.2)
            if i % 5 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-warn'>WARNING: Trabecular structural loss increasing.</span>")
        elif "Inhibitor Toxicity" in scenario and timeline_months > 7:
            hr += 30
            pgdh = np.random.uniform(0.1, 0.5)
            if i % 4 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-alert'>CRITICAL: ALT/AST elevated. Hepatotoxicity suspected.</span>")
        elif "Supply Chain" in scenario and timeline_months > 8:
            if i % 4 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-warn'>WARNING: 15-PGDH Inhibitor payload empty.</span>")
        elif "Cryogenic" in scenario:
            hr -= 15 # Bradycardia in extreme cold adaptation
            inf += 0.4 # Cold-induced inflammation
            if i % 4 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-warn'>WARNING: Cryo-induced joint fluid viscosity spiked.</span>")
        elif "Decompression" in scenario:
            hr += 20
            bone -= 0.5
            if i % 4 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-alert'>CRITICAL: Nitrogen micro-bubbles detected in synovial fluid.</span>")
        else:
            if i % 6 == 0: anomalies.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] <span class='console-info'>INFO: Nominal telemetry synced with Global Server.</span>")

        # Keep log short
        if len(anomalies) > 15: anomalies = anomalies[:15]

        current_telemetry = {
            'heart_rate': hr, 'bone_density': bone, 'radiation': rad,
            'pgdh': pgdh, 'inflammation_score': inf
        }
        
        rad_factor = rad / 1000.0
        bone_factor = abs(min(0, bone)) / 5.0
        
        # Deep space extremes degrade matrix faster
        env_factor = 1.5 if ("Titan" in node or "Europa" in node) else 1.0
        
        matrix_integrity = max(0.01, 1.0 - (rad_factor * env_factor) - bone_factor - (timeline_months/48))
        
        if "Supply Chain" in scenario and timeline_months > 8:
            matrix_integrity *= 0.6
            drug_dose = 0.0
        elif "Inhibitor Toxicity" in scenario and timeline_months > 7:
            drug_dose = 0.0
        else:
            drug_dose = min(50.0, pgdh * 2.5 + timeline_months)
            
        ode_trajectory = {"final_matrix_integrity": matrix_integrity, "recommended_drug_dose": drug_dose}
        risk_score = risk_model.predict_risk(current_telemetry)
        
        # --- UI Rendering ---
        metric_risk.metric("XGBoost Mission Risk", f"{risk_score:.2f}", delta=f"Month {timeline_months:.1f}")
        metric_integ.metric("Matrix Integrity Prediction", f"{matrix_integrity*100:.1f}%")
        metric_rad.metric("GCR/Env Stress Exposure", f"{rad:.1f}")
        
        radar_ph.plotly_chart(plot_radar_chart(current_telemetry, matrix_integrity), use_container_width=True, key=f"radar_{i}")
        
        log_html = "<div class='console-log'>" + "<br>".join(anomalies) + "</div>"
        log_ph.markdown("### 📡 Live Anomaly Log\n" + log_html, unsafe_allow_html=True)
        
        # --- Deep Inferences Updates ---
        if "Solar Particle" in scenario:
            vuln_desc.info("**Radiological Breakdown In-Depth:** Ionizing radiation from the SPE causes double-strand DNA breaks in chondrocytes. This triggers premature cellular senescence, marked by a massive upregulation of inflammatory cytokines (IL-1β, TNF-α) and the destructive enzyme MMP-13.")
        elif "Severe Microgravity" in scenario:
            vuln_desc.warning("**Biomechanical Breakdown In-Depth:** The absence of mechanical loading on the joint surface alters chondrocyte mechanotransduction. Without the cyclic pressure of gravity, the synthesis of Aggrecan and Type II collagen plummets.")
        elif "Inhibitor Toxicity" in scenario:
            vuln_desc.error("**Pharmacological Breakdown In-Depth:** While the 15-PGDH inhibitor rescues cartilage, over-suppression in this astronaut has induced severe hepatotoxicity. Liver enzymes are elevated. Drug must be aborted immediately.")
        elif "Cryogenic" in scenario:
            vuln_desc.warning("**Cryogenic Matrix Stiffening (Titan):** Ambient surface temperatures of -179°C cause the synovial fluid to increase in viscosity, leading to micro-tears in the articular cartilage under physical strain. Upregulated cold-shock proteins (RBM3) are insufficient to prevent matrix fragmentation.")
        elif "Decompression" in scenario:
            vuln_desc.error("**Submarine Decompression Sickness (Europa):** Rapid pressure changes during subsurface oceanic ops generate nitrogen micro-bubbles in the synovial fluid. This causes acute 'Bends' in the joints, mechanically rupturing the cartilage matrix and inducing rapid localized inflammation (IL-6 spike).")
        else:
            vuln_desc.success("**Nominal In-Depth:** Chondrocyte homeostasis is maintained. The 15-PGDH inhibitor is effectively suppressing MMP-13 activity, preserving the Type II collagen network.")

        # --- Generative Vision Updates ---
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        x = np.linspace(0, 255, 256); y = np.linspace(0, 255, 256); X, Y = np.meshgrid(x, y)
        heatmap = np.stack([np.full_like(X, int(255*(1-matrix_integrity))), Y, X], axis=-1).astype(np.uint8)
        future_img = (dummy_img * matrix_integrity).astype(np.uint8)
        
        v1_ph.image(dummy_img, caption="Baseline Structural SEM", use_column_width=True)
        v2_ph.image(heatmap, caption="Live Score-CAM Vulnerability Map", use_container_width=True)
        v3_ph.image(future_img, caption=f"Predicted Future (Month {timeline_months:.1f})", use_column_width=True)

        # --- Federated Learning Loss Curve Update ---
        # Slowly converging loss curve (always positive!)
        # e.g., starts at ~2.5, decays exponentially to ~0.08
        current_loss = 2.5 * np.exp(-0.15 * i) + np.random.uniform(0.01, 0.05) + 0.05
        loss_data.append(current_loss)
        
        # Plotly for live line chart
        fig_loss = go.Figure(data=go.Scatter(y=loss_data, mode='lines+markers', line=dict(color='#00ff00')))
        fig_loss.update_layout(
            title=f"Live Global Model Convergence (Epoch {i*42})",
            xaxis_title="Simulation Steps",
            yaxis_title="Cross-Entropy Loss",
            yaxis=dict(range=[0, 3.0]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            height=350
        )
        loss_ph.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{i}")

        # Update CMO rarely to save compute in simulation loop
        if i == 20 or i == 39:
            summary = cmo_agent.generate_cmo_summary(current_telemetry, risk_score, ode_trajectory)
            if "ERROR" in summary or summary == "":
                summary = f"**Live Update (Month {timeline_months:.1f}):** Matrix integrity is {matrix_integrity*100:.1f}%. Recommended dose: {drug_dose:.1f} mg. Risk: {risk_score:.2f}."
            cmo_ph.info(summary)
            
        time.sleep(0.3) # Simulation speed

    st.success("Simulation Complete. Final State Achieved. Model converged.")

if __name__ == "__main__":
    main()
