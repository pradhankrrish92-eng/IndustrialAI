from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Industrial AI | Telemetry Diagnostics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom colorful CSS styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
    }
    .badge-ok {
        background-color: #059669;
        color: white;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-danger {
        background-color: #dc2626;
        color: white;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("artifacts/multilabel_model.joblib")


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

if pipeline is None:
    st.error(
        f"Model artifact not found at `{MODEL_PATH}`. Please run `python scripts/train_multilabel.py` first."
    )
    st.stop()

# Human-friendly dictionary mapping
FAILURE_DESCRIPTIONS = {
    "twf": {
        "title": "🔪 Tool Wear Failure",
        "desc": "The cutting tool edge has exceeded safe wear limits and needs replacement.",
        "color": "#f59e0b",
    },
    "hdf": {
        "title": "🔥 Heat Dissipation Failure",
        "desc": "The temperature gap between process and environment is causing cooling failure.",
        "color": "#ef4444",
    },
    "pwf": {
        "title": "⚡ Power Supply & Electrical Failure",
        "desc": "Torque and rotational speed combination produces dangerous power draw.",
        "color": "#ec4899",
    },
    "osf": {
        "title": "🔩 Mechanical Overstrain Failure",
        "desc": "Excessive torque under worn tool conditions exceeds material yield strength.",
        "color": "#8b5cf6",
    },
    "rnf": {
        "title": "🎲 Random Sensor / Uncorrelated Fault",
        "desc": "Anomalous transient spike outside known standard physical mechanics.",
        "color": "#06b6d4",
    },
}

# Sidebar: Controls with direct typing inputs
st.sidebar.title("🎛️ Sensor Control Panel")
st.sidebar.caption("Adjust or type values directly into the input fields.")

machine_type = st.sidebar.selectbox(
    "Product Quality Grade",
    options=["L", "M", "H"],
    index=0,
    help="L: Low variant (50%), M: Medium (30%), H: Heavy/High variant (20%)",
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🌡️ Thermal Parameters")
air_temp = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=280.0,
    max_value=320.0,
    value=298.2,
    step=0.2,
    format="%.2f",
)
process_temp = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=290.0,
    max_value=330.0,
    value=308.7,
    step=0.2,
    format="%.2f",
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Kinematic & Tool Parameters")
rot_speed = st.sidebar.number_input(
    "Rotational Speed [RPM]",
    min_value=1000.0,
    max_value=3200.0,
    value=1500.0,
    step=25.0,
    format="%.1f",
)
torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=2.0,
    max_value=90.0,
    value=42.0,
    step=1.0,
    format="%.1f",
)
tool_wear = st.sidebar.number_input(
    "Tool Wear [minutes]",
    min_value=0.0,
    max_value=300.0,
    value=35.0,
    step=5.0,
    format="%.1f",
)

# Inference dataframe
input_df = pd.DataFrame(
    [
        {
            "type": machine_type,
            "air_temperature_k": air_temp,
            "process_temperature_k": process_temp,
            "rotational_speed_rpm": rot_speed,
            "torque_nm": torque,
            "tool_wear_min": tool_wear,
        }
    ]
)

# Model Prediction
raw_preds = pipeline.predict(input_df)[0]
is_failure = bool(raw_preds[0])
target_keys = ["machine_failure", "twf", "hdf", "pwf", "osf", "rnf"]

# Top Header Status Section
st.title("🏭 Predictive Maintenance & Fault Diagnostics")
st.markdown("Live equipment health telemetry simulator powered by Multi-Label Random Forest.")

status_col1, status_col2 = st.columns([1, 2])

with status_col1:
    if is_failure:
        st.markdown(
            '<div class="badge-danger">⚠️ BREAKDOWN IMMINENT</div>',
            unsafe_allow_html=True,
        )
        st.error("Telemetry indicates one or more operational parameters are in critical state.")
    else:
        st.markdown(
            '<div class="badge-ok">✅ SYSTEM HEALTH: OPTIMAL</div>',
            unsafe_allow_html=True,
        )
        st.success("All sensor metrics are within nominal design thresholds.")

with status_col2:
    flagged_keys = [
        target_keys[i]
        for i, val in enumerate(raw_preds[1:], start=1)
        if val == 1
    ]
    if flagged_keys:
        st.subheader("🚨 Diagnosed Failure Root Causes:")
        for key in flagged_keys:
            info = FAILURE_DESCRIPTIONS[key]
            st.markdown(
                f"- **{info['title']}**: *{info['desc']}*"
            )
    else:
        st.subheader("🛡️ Root Cause Analysis:")
        st.info("No mechanical, power, or thermal failure mechanisms triggered.")

# Physical indicators
delta_t = process_temp - air_temp
power_w = torque * (rot_speed * 2.0 * np.pi / 60.0)
overstrain = tool_wear * torque

st.markdown("---")
st.subheader("⚡ Real-Time Derived Physical Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Thermal Delta (ΔT)", f"{delta_t:.2f} K", delta="Danger > 11.0 K" if delta_t >= 11.0 else "Normal")
m2.metric("Mechanical Power", f"{power_w:.1f} W", delta="Surge > 7000 W" if power_w > 7000 else "Stable")
m3.metric("Overstrain Index", f"{overstrain:.1f}", delta="Risk > 11000" if overstrain > 11000 else "Safe")
m4.metric("Machine Variant", f"Grade {machine_type}", delta="Selected Profile")

st.markdown("---")

# Visual Charts Section
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📊 Diagnostic Alert Severity")
    diagnostics_data = []
    for key in ["twf", "hdf", "pwf", "osf", "rnf"]:
        is_active = 1 if key in flagged_keys else 0
        diagnostics_data.append(
            {
                "Fault Category": FAILURE_DESCRIPTIONS[key]["title"],
                "Alert Level": is_active,
                "Color": FAILURE_DESCRIPTIONS[key]["color"] if is_active else "#334155",
            }
        )
    df_chart = pd.DataFrame(diagnostics_data)

    fig_bar = px.bar(
        df_chart,
        x="Alert Level",
        y="Fault Category",
        orientation="h",
        color="Fault Category",
        color_discrete_map={
            row["Fault Category"]: row["Color"] for _, row in df_chart.iterrows()
        },
        range_x=[0, 1.2],
    )
    fig_bar.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        xaxis=dict(showgrid=False, tickvals=[0, 1], ticktext=["Clear (0)", "Triggered (1)"]),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    st.subheader("🎯 Operating Envelope (Torque vs. Speed)")
    # Scatter plot showing where the machine is currently operating
    fig_scatter = go.Figure()

    # Safe operating zone reference (elliptical boundary approximation)
    theta = np.linspace(0, 2 * np.pi, 100)
    safe_speed = 1500 + 400 * np.cos(theta)
    safe_torque = 40 + 20 * np.sin(theta)
    
    fig_scatter.add_trace(
        go.Scatter(
            x=safe_speed,
            y=safe_torque,
            mode="lines",
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.15)",
            line=dict(color="#10b981", dash="dot"),
            name="Normal Envelope",
        )
    )

    # Current machine telemetry coordinate
    fig_scatter.add_trace(
        go.Scatter(
            x=[rot_speed],
            y=[torque],
            mode="markers+text",
            marker=dict(
                size=18,
                color="#ef4444" if is_failure else "#10b981",
                line=dict(width=3, color="#ffffff"),
            ),
            text=["📍 Current State"],
            textposition="top center",
            name="Operating Telemetry",
        )
    )

    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        xaxis=dict(title="Rotational Speed [RPM]", range=[900, 3100], showgrid=True, gridcolor="#334155"),
        yaxis=dict(title="Torque [Nm]", range=[0, 90], showgrid=True, gridcolor="#334155"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)