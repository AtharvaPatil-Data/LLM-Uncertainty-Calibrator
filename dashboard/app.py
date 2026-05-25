# =============================================================================
# LLM-Uncertainty-Calibrator — Premium Streamlit Dashboard
# Run: streamlit run dashboard/app.py
# =============================================================================

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
from plots import (
    plot_reliability_diagram,
    plot_ece_comparison,
    plot_conformal_sets,
    plot_probability_distributions,
    plot_probability_heatmap,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uncertainty Calibrator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
        background: #0a0e27;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero header with gradient */
    .hero-header {
        background: linear-gradient(135deg, #0a0e27 0%, #1a2238 50%, #151b3d 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 50%, rgba(100, 255, 218, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .hero-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #64ffda;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero-header p {
        color: #8892b0;
        font-size: 1.1rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    /* Metric cards with glassmorphism */
    .metric-card {
        background: rgba(26, 34, 56, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #64ffda;
        box-shadow: 0 8px 24px rgba(100, 255, 218, 0.12);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Color schemes for metrics */
    .metric-danger { color: #ff5370; }
    .metric-success { color: #c3e88d; }
    .metric-primary { color: #64ffda; }
    .metric-secondary { color: #c792ea; }
    .metric-warning { color: #ffcb6b; }
    .metric-blue { color: #82aaff; }
    
    /* Section headers with accent bar */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8892b0;
        border-left: 4px solid #64ffda;
        padding-left: 1rem;
        margin: 2.5rem 0 1.5rem 0;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, rgba(100, 255, 218, 0.08) 0%, rgba(130, 170, 255, 0.06) 100%);
        border: 1px solid rgba(100, 255, 218, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-card h4 {
        color: #64ffda;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .info-card p {
        color: #e4e9f7;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Sample explorer cards */
    .sample-card {
        background: #151b3d;
        border: 1px solid #2d3561;
        border-left: 3px solid #64ffda;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .sample-text {
        color: #e4e9f7;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-top: 1px solid #2d3561;
    }
    
    .prob-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .prob-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-correct { background: rgba(195, 232, 141, 0.15); color: #c3e88d; }
    .badge-incorrect { background: rgba(255, 83, 112, 0.15); color: #ff5370; }
    .badge-low { background: rgba(100, 255, 218, 0.15); color: #64ffda; }
    .badge-medium { background: rgba(255, 203, 107, 0.15); color: #ffcb6b; }
    .badge-high { background: rgba(255, 83, 112, 0.15); color: #ff5370; }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0a0e27;
        border-right: 1px solid #2d3561;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #151b3d;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #64ffda;
        color: #0a0e27;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "results.json")

@st.cache_data
def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def load_or_demo() -> tuple[dict, bool]:
    """Return (results_dict, is_demo)."""
    if os.path.exists(DATA_PATH):
        return load_results(DATA_PATH), False
    else:
        st.error("⚠️ **results.json not found.** Run colab_inference.py first.")
        st.stop()

# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    results, is_demo = load_or_demo()
    meta = results["metadata"]
    baseline = results["baseline"]
    temp = results["temperature_scaling"]
    platt = results["platt_scaling"]
    conf = results["conformal_prediction"]
    
    # ── Hero Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>🎯 Uncertainty Calibrator</h1>
        <p>Statistical Calibration & Conformal Prediction for LLM Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Experiment Configuration")
        st.markdown(f"**Model:** `{meta['model']}`")
        st.markdown(f"**Classes:** {meta['num_classes']}")
        st.markdown(f"**Test Samples:** {meta['test_size']}")
        st.markdown(f"**Conformal α:** {meta['conf_alpha']}")
        
        st.divider()
        
        st.markdown("### 📊 About")
        st.markdown(
            "This dashboard evaluates **uncertainty calibration** for financial risk classification. "
            "Uncalibrated models are overconfident — saying '95% sure' when they should say '70% sure'. "
            "We apply three calibration methods and measure improvement using **Expected Calibration Error (ECE)**."
        )
        
        st.divider()
        
        st.markdown("### 🔬 Methods")
        st.markdown("**Temperature Scaling** — Single parameter T")
        st.markdown("**Platt Scaling** — Logistic regression")
        st.markdown("**Conformal Prediction** — Prediction sets with coverage guarantees")
    
    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Calibration Performance</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Baseline ECE
    c1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline ECE</div>
        <div class="metric-value metric-danger">{baseline['ece']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Best calibration
    best_ece = min(temp['ece'], platt['ece'])
    best_reduction = max(temp['ece_reduction'], platt['ece_reduction'])
    improvement_pct = (baseline['ece'] - best_ece) / baseline['ece'] * 100
    
    c2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best Calibrated ECE</div>
        <div class="metric-value metric-success">{best_ece:.4f}</div>
        <div class="metric-delta metric-success">↓ {improvement_pct:.1f}% improvement</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Conformal coverage
    coverage_diff = abs(conf['coverage'] - conf['target_coverage'])
    coverage_color = "metric-success" if coverage_diff < 0.05 else "metric-warning"
    
    c3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Conformal Coverage</div>
        <div class="metric-value {coverage_color}">{conf['coverage']:.1%}</div>
        <div class="metric-delta metric-blue">Target: {conf['target_coverage']:.0%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Avg prediction set size
    c4.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Set Size</div>
        <div class="metric-value metric-primary">{conf['avg_set_size']:.2f}</div>
        <div class="metric-delta metric-blue">classes per prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Visualization Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Reliability Analysis",
        "🎯 Calibration Comparison", 
        "🔮 Conformal Prediction",
        "🔍 Sample Explorer"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Reliability Diagram</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>How to Read This Chart</h4>
            <p>
            Points closer to the diagonal line = better calibration. 
            If a model says "80% confident", it should be correct 80% of the time.
            Marker size = number of samples in that confidence bin.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_reliability = plot_reliability_diagram(results)
        st.plotly_chart(fig_reliability, use_container_width=True)
        
        st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
        fig_dist = plot_probability_distributions(results)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header">ECE Comparison</div>', unsafe_allow_html=True)
        
        fig_ece = plot_ece_comparison(results)
        st.plotly_chart(fig_ece, use_container_width=True)
        
        # Method comparison table
        st.markdown('<div class="section-header">Method Details</div>', unsafe_allow_html=True)
        
        comparison_data = {
            "Method": ["Baseline", "Temperature Scaling", "Platt Scaling"],
            "ECE": [f"{baseline['ece']:.4f}", f"{temp['ece']:.4f}", f"{platt['ece']:.4f}"],
            "Reduction": ["—", f"{temp['ece_reduction']:.4f}", f"{platt['ece_reduction']:.4f}"],
            "Improvement": ["—", f"{temp['ece_reduction']/baseline['ece']*100:.1f}%", f"{platt['ece_reduction']/baseline['ece']*100:.1f}%"],
        }
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Temperature parameter
        st.markdown(f"""
        <div class="info-card">
            <h4>Temperature Scaling Parameter</h4>
            <p>Optimal T = <strong>{temp['optimal_temperature']:.4f}</strong></p>
            <p style="margin-top: 0.5rem; font-size: 0.85rem;">
            T > 1.0 means the model was overconfident (most common). 
            T < 1.0 would mean underconfident (rare).
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="section-header">Prediction Set Size Distribution</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card">
            <h4>Conformal Prediction Guarantees</h4>
            <p>
            With 90% coverage target, the true label should be in the prediction set 
            <strong>at least 90% of the time</strong>. Achieved: <strong>{conf['coverage']:.1%}</strong>.
            Average set size: <strong>{conf['avg_set_size']:.2f}</strong> classes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_conf = plot_conformal_sets(results)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Show some examples
        st.markdown('<div class="section-header">Example Prediction Sets</div>', unsafe_allow_html=True)
        
        samples = results["test_samples"]
        label_names = ["Low Risk", "Medium Risk", "High Risk"]
        
        for i in range(min(5, len(samples))):
            s = samples[i]
            true_label = label_names[s["true_label"]]
            pred_set = [label_names[idx] for idx in s["conformal_pred_set"]]
            correct = s["true_label"] in s["conformal_pred_set"]
            badge_class = "badge-correct" if correct else "badge-incorrect"
            badge_text = "✓ Covered" if correct else "✗ Missed"
            
            st.markdown(f"""
            <div class="sample-card">
                <div class="sample-text">"{s['text'][:150]}..."</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge badge-{true_label.lower().replace(' ', '-')}">{true_label}</span>
                        <span style="color: #8892b0; margin: 0 0.5rem;">→</span>
                        <span style="color: #e4e9f7; font-family: 'JetBrains Mono', monospace;">
                            {{{', '.join(pred_set)}}}
                        </span>
                    </div>
                    <span class="badge {badge_class}">{badge_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header">Individual Sample Analysis</div>', unsafe_allow_html=True)
        
        # Sample selector
        sample_idx = st.slider("Select Sample", 0, len(results["test_samples"]) - 1, 0)
        s = results["test_samples"][sample_idx]
        label_names = ["Low Risk", "Medium Risk", "High Risk"]
        
        # Display text
        true_label = label_names[s["true_label"]]
        st.markdown(f"""
        <div class="sample-card">
            <div class="sample-text">"{s['text']}"</div>
            <span class="badge badge-{true_label.lower().replace(' ', '-')}">True: {true_label}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability comparison
        st.markdown('<div class="section-header">Probability Comparison</div>', unsafe_allow_html=True)
        
        prob_df = pd.DataFrame({
            "Class": label_names,
            "Uncalibrated": s["probs_uncalibrated"],
            "Temperature": s["probs_temp_scaled"],
            "Platt": s["probs_platt_scaled"],
        })
        
        st.dataframe(
            prob_df.style.background_gradient(subset=["Uncalibrated", "Temperature", "Platt"], cmap="viridis"),
            use_container_width=True,
            hide_index=True,
        )
        
        # Prediction set
        pred_set_names = [label_names[i] for i in s["conformal_pred_set"]]
        st.markdown(f"""
        <div class="info-card">
            <h4>Conformal Prediction Set</h4>
            <p>{', '.join(pred_set_names)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #8892b0; font-size: 0.85rem;'>"
        f"Uncertainty Calibrator | Evaluated {meta['test_size']} samples | "
        f"<code style='color: #64ffda;'>{meta['timestamp']}</code>"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
