# =============================================================================
# LLM-Uncertainty-Calibrator — Premium Plotly Visualizations
# =============================================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Premium Design System ─────────────────────────────────────────────────────
# Dark theme with vibrant accents, inspired by modern ML platforms

BG = "#0a0e27"
PAPER_BG = "#151b3d"
CARD_BG = "#1a2238"
GRID = "#2d3561"
TEXT = "#e4e9f7"
MUTED = "#8892b0"
PRIMARY = "#64ffda"      # Teal/cyan accent
SECONDARY = "#c792ea"    # Purple
DANGER = "#ff5370"       # Coral red
WARNING = "#ffcb6b"      # Gold
SUCCESS = "#c3e88d"      # Green
BLUE = "#82aaff"         # Sky blue

FONT_MONO = "JetBrains Mono, Fira Code, Consolas, monospace"
FONT_SANS = "Inter, -apple-system, system-ui, sans-serif"

def _base_layout(**kwargs):
    """Premium base layout with glassmorphism effects."""
    layout = dict(
        plot_bgcolor=BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_SANS, color=TEXT, size=12),
        margin=dict(l=60, r=40, t=60, b=50),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor=CARD_BG,
            font_size=11,
            font_family=FONT_MONO,
            bordercolor=PRIMARY,
        ),
    )
    layout.update(kwargs)
    return layout


# ── 1. Reliability Diagram (Confidence vs Accuracy) ──────────────────────────

def plot_reliability_diagram(results: dict, n_bins: int = 10) -> go.Figure:
    """
    Reliability diagram comparing uncalibrated vs calibrated probabilities.
    Shows how well confidence aligns with actual accuracy.
    """
    samples = results["test_samples"]
    
    # Extract data
    true_labels = np.array([s["true_label"] for s in samples])
    probs_uncal = np.array([s["probs_uncalibrated"] for s in samples])
    probs_temp = np.array([s["probs_temp_scaled"] for s in samples])
    probs_platt = np.array([s["probs_platt_scaled"] for s in samples])
    
    preds_uncal = np.argmax(probs_uncal, axis=1)
    preds_temp = np.argmax(probs_temp, axis=1)
    preds_platt = np.argmax(probs_platt, axis=1)
    
    conf_uncal = np.max(probs_uncal, axis=1)
    conf_temp = np.max(probs_temp, axis=1)
    conf_platt = np.max(probs_platt, axis=1)
    
    def compute_bins(confidences, predictions, labels, n_bins):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accs, confs, counts = [], [], []
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            if in_bin.sum() > 0:
                acc = (predictions[in_bin] == labels[in_bin]).mean()
                conf = confidences[in_bin].mean()
                count = in_bin.sum()
            else:
                acc, conf, count = 0, (lower + upper) / 2, 0
            accs.append(acc)
            confs.append(conf)
            counts.append(count)
        return np.array(accs), np.array(confs), np.array(counts)
    
    accs_u, confs_u, counts_u = compute_bins(conf_uncal, preds_uncal, true_labels, n_bins)
    accs_t, confs_t, counts_t = compute_bins(conf_temp, preds_temp, true_labels, n_bins)
    accs_p, confs_p, counts_p = compute_bins(conf_platt, preds_platt, true_labels, n_bins)
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color=MUTED, width=2, dash='dash'),
        hoverinfo='skip',
    ))
    
    # Uncalibrated
    fig.add_trace(go.Scatter(
        x=confs_u, y=accs_u,
        mode='markers+lines',
        name='Uncalibrated',
        marker=dict(
            size=counts_u * 2,  # Size = sample count
            color=DANGER,
            opacity=0.8,
            line=dict(width=2, color=BG),
        ),
        line=dict(color=DANGER, width=2),
        hovertemplate='<b>Uncalibrated</b><br>Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br><extra></extra>',
    ))
    
    # Temperature Scaled
    fig.add_trace(go.Scatter(
        x=confs_t, y=accs_t,
        mode='markers+lines',
        name='Temperature Scaling',
        marker=dict(
            size=counts_t * 2,
            color=PRIMARY,
            opacity=0.8,
            line=dict(width=2, color=BG),
        ),
        line=dict(color=PRIMARY, width=2),
        hovertemplate='<b>Temperature Scaling</b><br>Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br><extra></extra>',
    ))
    
    # Platt Scaled
    fig.add_trace(go.Scatter(
        x=confs_p, y=accs_p,
        mode='markers+lines',
        name='Platt Scaling',
        marker=dict(
            size=counts_p * 2,
            color=SECONDARY,
            opacity=0.8,
            line=dict(width=2, color=BG),
        ),
        line=dict(color=SECONDARY, width=2),
        hovertemplate='<b>Platt Scaling</b><br>Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br><extra></extra>',
    ))
    
    fig.update_layout(
        **_base_layout(
            title=dict(
                text='Reliability Diagram: Confidence vs Accuracy',
                font=dict(size=18, family=FONT_SANS, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='Confidence',
                range=[0, 1],
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED, family=FONT_MONO),
                title_font=dict(color=TEXT, size=13),
            ),
            yaxis=dict(
                title='Accuracy',
                range=[0, 1],
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED, family=FONT_MONO),
                title_font=dict(color=TEXT, size=13),
            ),
            legend=dict(
                bgcolor=CARD_BG,
                bordercolor=GRID,
                borderwidth=1,
                font=dict(size=11, family=FONT_SANS),
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
            ),
        )
    )
    
    return fig


# ── 2. ECE Comparison Bar Chart ──────────────────────────────────────────────

def plot_ece_comparison(results: dict) -> go.Figure:
    """Horizontal bar chart showing ECE before and after calibration."""
    methods = ['Baseline', 'Temperature\nScaling', 'Platt\nScaling']
    eces = [
        results["baseline"]["ece"],
        results["temperature_scaling"]["ece"],
        results["platt_scaling"]["ece"],
    ]
    colors = [DANGER, PRIMARY, SECONDARY]
    
    # Add percentage reductions as annotations
    reductions = [
        None,
        f"↓ {results['temperature_scaling']['ece_reduction']:.4f}",
        f"↓ {results['platt_scaling']['ece_reduction']:.4f}" if results['platt_scaling']['ece_reduction'] > 0 else f"↑ {abs(results['platt_scaling']['ece_reduction']):.4f}",
    ]
    
    fig = go.Figure()
    
    for i, (method, ece, color) in enumerate(zip(methods, eces, colors)):
        fig.add_trace(go.Bar(
            y=[method],
            x=[ece],
            orientation='h',
            name=method,
            marker=dict(
                color=color,
                opacity=0.85,
                line=dict(color=color, width=2),
            ),
            text=[f'{ece:.4f}'],
            textposition='outside',
            textfont=dict(size=14, family=FONT_MONO, color=color),
            hovertemplate=f'<b>{method}</b><br>ECE: %{{x:.4f}}<extra></extra>',
        ))
    
    fig.update_layout(
        **_base_layout(
            title=dict(
                text='Expected Calibration Error (ECE) — Lower is Better',
                font=dict(size=18, family=FONT_SANS, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='ECE',
                range=[0, max(eces) * 1.3],
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED, family=FONT_MONO),
                title_font=dict(color=TEXT, size=13),
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color=TEXT, size=12, family=FONT_SANS),
            ),
            showlegend=False,
            height=300,
        )
    )
    
    return fig


# ── 3. Conformal Prediction Set Size Distribution ────────────────────────────

def plot_conformal_sets(results: dict) -> go.Figure:
    """Distribution of prediction set sizes from Conformal Prediction."""
    samples = results["test_samples"]
    set_sizes = [len(s["conformal_pred_set"]) for s in samples]
    
    # Count distribution
    unique_sizes, counts = np.unique(set_sizes, return_counts=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f'{int(s)} classes' for s in unique_sizes],
        y=counts,
        marker=dict(
            color=PRIMARY,
            opacity=0.8,
            line=dict(color=PRIMARY, width=2),
        ),
        text=counts,
        textposition='outside',
        textfont=dict(size=14, family=FONT_MONO, color=PRIMARY),
        hovertemplate='<b>Set Size:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>',
    ))
    
    avg_size = results["conformal_prediction"]["avg_set_size"]
    coverage = results["conformal_prediction"]["coverage"]
    
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f'Conformal Prediction Set Sizes<br><sub style="color:{MUTED}">Avg: {avg_size:.2f} | Coverage: {coverage:.1%}</sub>',
                font=dict(size=18, family=FONT_SANS, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='Prediction Set Size',
                showgrid=False,
                tickfont=dict(color=TEXT, size=12, family=FONT_SANS),
                title_font=dict(color=TEXT, size=13),
            ),
            yaxis=dict(
                title='Frequency',
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED, family=FONT_MONO),
                title_font=dict(color=TEXT, size=13),
            ),
            showlegend=False,
            height=350,
        )
    )
    
    return fig


# ── 4. Probability Distribution Comparison ───────────────────────────────────

def plot_probability_distributions(results: dict) -> go.Figure:
    """
    Shows distribution of predicted probabilities across all test samples.
    Helps visualize overconfidence vs well-calibrated distributions.
    """
    samples = results["test_samples"]
    
    # Get max probabilities from each method
    max_probs_uncal = [max(s["probs_uncalibrated"]) for s in samples]
    max_probs_temp = [max(s["probs_temp_scaled"]) for s in samples]
    max_probs_platt = [max(s["probs_platt_scaled"]) for s in samples]
    
    fig = go.Figure()
    
    # Uncalibrated
    fig.add_trace(go.Violin(
        y=max_probs_uncal,
        name='Uncalibrated',
        side='negative',
        line_color=DANGER,
        fillcolor=DANGER,
        opacity=0.6,
        meanline_visible=True,
        points='outliers',
        pointpos=-0.5,
        marker=dict(size=4, opacity=0.5),
    ))
    
    # Temperature Scaled
    fig.add_trace(go.Violin(
        y=max_probs_temp,
        name='Temperature',
        side='positive',
        line_color=PRIMARY,
        fillcolor=PRIMARY,
        opacity=0.6,
        meanline_visible=True,
        points='outliers',
        pointpos=0.5,
        marker=dict(size=4, opacity=0.5),
    ))
    
    fig.update_layout(
        **_base_layout(
            title=dict(
                text='Confidence Distribution: Uncalibrated vs Calibrated',
                font=dict(size=18, family=FONT_SANS, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
            yaxis=dict(
                title='Max Probability',
                range=[0, 1],
                showgrid=True,
                gridcolor=GRID,
                tickfont=dict(color=MUTED, family=FONT_MONO),
                title_font=dict(color=TEXT, size=13),
            ),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
            ),
            violinmode='overlay',
            legend=dict(
                bgcolor=CARD_BG,
                bordercolor=GRID,
                borderwidth=1,
                font=dict(size=11, family=FONT_SANS),
            ),
            height=400,
        )
    )
    
    return fig


# ── 5. Interactive Heatmap: Probability Matrix ───────────────────────────────

def plot_probability_heatmap(results: dict, method: str = 'uncalibrated') -> go.Figure:
    """
    Heatmap showing predicted probabilities for each test sample across all classes.
    Rows = samples, Columns = classes, Color = probability
    """
    samples = results["test_samples"]
    
    prob_key = {
        'uncalibrated': 'probs_uncalibrated',
        'temperature': 'probs_temp_scaled',
        'platt': 'probs_platt_scaled',
    }[method]
    
    probs = np.array([s[prob_key] for s in samples])
    true_labels = np.array([s["true_label"] for s in samples])
    
    # Sort by true label for better visualization
    sorted_idx = np.argsort(true_labels)
    probs_sorted = probs[sorted_idx]
    labels_sorted = true_labels[sorted_idx]
    
    fig = go.Figure(data=go.Heatmap(
        z=probs_sorted,
        x=['Low Risk', 'Medium Risk', 'High Risk'],
        y=[f'Sample {i+1}' for i in range(len(samples))],
        colorscale=[
            [0.0, BG],
            [0.3, CARD_BG],
            [0.5, BLUE],
            [0.7, WARNING],
            [1.0, DANGER],
        ],
        colorbar=dict(
            title='Probability',
            titleside='right',
            tickmode='linear',
            tick0=0,
            dtick=0.2,
            tickfont=dict(color=MUTED, family=FONT_MONO),
            title_font=dict(color=TEXT),
        ),
        hovertemplate='Sample: %{y}<br>Class: %{x}<br>Prob: %{z:.3f}<extra></extra>',
    ))
    
    method_name = method.replace('_', ' ').title()
    
    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f'Probability Matrix — {method_name}',
                font=dict(size=18, family=FONT_SANS, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                side='top',
                tickfont=dict(color=TEXT, size=12, family=FONT_SANS),
            ),
            yaxis=dict(
                autorange='reversed',
                showticklabels=False,  # Too many samples
                title='Test Samples (sorted by true label)',
                title_font=dict(color=TEXT, size=11),
            ),
            height=600,
        )
    )
    
    return fig
