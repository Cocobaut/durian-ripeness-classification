"""Reusable Streamlit UI sections, styling, input widgets, and Plotly charts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.datasets.image_dataset import IMAGE_SIZE, camera_angle_from_name, fruit_id_from_name
from src.utils.common import CLASS_ORDER
from visualization.config import (
    AGG_MAJORITY,
    AGG_MEAN_PROB,
    CLASS_COLOR_MAP,
    CLASS_METADATA,
    DEFAULT_BEST_CKPT,
    DEFAULT_LAST_CKPT,
    MODE_MULTI,
    MODE_SINGLE,
    SUPPORTED_IMAGE_TYPES,
)
from visualization.inference import (
    aggregate_fruit_predictions,
    open_rgb_image,
    save_uploaded_checkpoint,
)


def inject_custom_css() -> None:
    """Inject clean styling for the header banner and ripeness verdict card."""
    st.markdown(
        """
        <style>
        .hero-header {
            padding: 1.1rem 1.35rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(34, 197, 94, 0.10) 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            margin-bottom: 1.1rem;
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            opacity: 0.85;
            margin: 0;
        }
        .verdict-card {
            padding: 1.2rem 1.35rem;
            border-radius: 14px;
            margin: 0.75rem 0 1rem 0;
        }
        .verdict-badge {
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
        }
        .verdict-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0.1rem 0 0.35rem 0;
            line-height: 1.15;
        }
        .verdict-desc {
            font-size: 0.94rem;
            margin: 0.22rem 0;
            opacity: 0.92;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the top dashboard banner."""
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">🍈 Durian Ripeness Classification Dashboard</div>
            <p class="hero-subtitle">
                Vision-Only Student Model (CMKD Stage 2) &bull; Standard pipeline:
                <code>get_eval_transforms(384)</code> &rarr; <code>VisionStudent</code> &rarr;
                <strong>Unripe</strong> / <strong>Ripe</strong> / <strong>Overripe</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls() -> Tuple[Optional[str], str]:
    """Render checkpoint and compute device selectors in the sidebar."""
    with st.sidebar:
        st.header("Student Model Settings")
        ckpt_options: List[str] = []
        if DEFAULT_BEST_CKPT.is_file():
            ckpt_options.append("Best Student (best_student.pth)")
        if DEFAULT_LAST_CKPT.is_file():
            ckpt_options.append("Last Student (last_student.pth)")
        ckpt_options.append("Upload Custom Checkpoint (.pth)")

        ckpt_choice = st.selectbox("Checkpoint", options=ckpt_options, index=0)
        resolved_ckpt_path: Optional[str] = None
        if ckpt_choice.startswith("Best Student"):
            resolved_ckpt_path = str(DEFAULT_BEST_CKPT)
        elif ckpt_choice.startswith("Last Student"):
            resolved_ckpt_path = str(DEFAULT_LAST_CKPT)
        else:
            custom_ckpt = st.file_uploader("Upload Student .pth", type=["pth", "pt"], key="custom_ckpt_upload")
            if custom_ckpt is not None:
                resolved_ckpt_path = save_uploaded_checkpoint(custom_ckpt)

        available_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device_str = st.selectbox("Compute Device", options=available_devices, index=0)

    return resolved_ckpt_path, device_str


def render_sidebar_model_summary(ckpt_meta: Dict[str, Any], device_str: str) -> None:
    """Display loaded model architecture and checkpoint metadata in the sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("Model Summary")
        s_cfg = ckpt_meta.get("student_config", {})
        st.markdown(
            f"""
            - **Checkpoint:** `{ckpt_meta['ckpt_name']}`
            - **Backbone:** `{s_cfg.get('backbone_name', 'convnextv2_tiny')}`
            - **Input Size:** `{IMAGE_SIZE}×{IMAGE_SIZE} RGB`
            - **Saved Epoch:** `{ckpt_meta['epoch']}` (`{ckpt_meta['phase']}`)
            - **Parameters:** `{ckpt_meta['num_params'] / 1e6:.2f}M` (`{ckpt_meta['file_size_mb']:.1f} MB`)
            - **Device:** `{device_str.upper()}`
            """
        )


def render_mode_selector() -> Tuple[str, str]:
    """Render prediction mode and (for multi-view) aggregation strategy selectors."""
    top_col1, top_col2 = st.columns([1.3, 1.0])
    with top_col1:
        prediction_mode = st.radio(
            "Prediction Mode",
            options=[MODE_SINGLE, MODE_MULTI],
            horizontal=True,
        )
    with top_col2:
        agg_strategy = AGG_MEAN_PROB
        if prediction_mode == MODE_MULTI:
            agg_strategy = st.radio(
                "Fruit-Level Aggregation Strategy",
                options=[AGG_MEAN_PROB, AGG_MAJORITY],
                horizontal=True,
            )
    return prediction_mode, agg_strategy


def collect_input_items(prediction_mode: str) -> List[Dict[str, Any]]:
    """Collect either a single RGB image or multiple angle images via upload or camera."""
    items: List[Dict[str, Any]] = []

    if prediction_mode == MODE_SINGLE:
        input_source = st.radio(
            "Input Source",
            options=["Upload Image File", "Camera Capture"],
            horizontal=True,
            key="single_input_source",
        )
        if input_source == "Upload Image File":
            uploaded = st.file_uploader(
                "Upload 1 durian RGB image (JPG, JPEG, PNG, WEBP, BMP)",
                type=list(SUPPORTED_IMAGE_TYPES),
                accept_multiple_files=False,
                key="single_file_uploader",
            )
            if uploaded is not None:
                items.append({"name": uploaded.name, "bytes": uploaded.getvalue()})
        else:
            captured = st.camera_input("Capture 1 durian photo", key="single_camera_input")
            if captured is not None:
                items.append({"name": captured.name or "camera_capture.jpg", "bytes": captured.getvalue()})
        return items

    if "captured_views" not in st.session_state:
        st.session_state["captured_views"] = []
    if "cam_counter" not in st.session_state:
        st.session_state["cam_counter"] = 0

    input_source = st.radio(
        "Input Source",
        options=["Upload Multiple Images", "Capture Multiple Angles (Camera)"],
        horizontal=True,
        key="multi_input_source",
    )

    if input_source == "Upload Multiple Images":
        uploaded_list = st.file_uploader(
            "Upload multiple RGB images of the same durian fruit (e.g., TOP, BOT, SA, SB, SC, SD)",
            type=list(SUPPORTED_IMAGE_TYPES),
            accept_multiple_files=True,
            key="multi_file_uploader",
        )
        if uploaded_list:
            for f in uploaded_list:
                items.append({"name": f.name, "bytes": f.getvalue()})
        return items

    cam_shot = st.camera_input(
        "Take a photo of one angle, then click 'Add Photo to Fruit Views'",
        key=f"multi_camera_{st.session_state['cam_counter']}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add Photo to Fruit Views", disabled=cam_shot is None, width="stretch"):
            idx = len(st.session_state["captured_views"]) + 1
            st.session_state["captured_views"].append(
                {"name": f"camera_view_{idx}.jpg", "bytes": cam_shot.getvalue()}
            )
            st.session_state["cam_counter"] += 1
            st.rerun()
    with c2:
        if st.button(
            "Clear Captured Views",
            disabled=len(st.session_state["captured_views"]) == 0,
            width="stretch",
        ):
            st.session_state["captured_views"] = []
            st.session_state["last_prediction"] = None
            st.rerun()

    items.extend(st.session_state["captured_views"])
    return items


def render_preview_and_trigger(prediction_mode: str, items: List[Dict[str, Any]]) -> bool:
    """Display uploaded/captured RGB previews and render the Run Ripeness Prediction button."""
    if prediction_mode == MODE_SINGLE:
        preview_rgb = open_rgb_image(items[0]["bytes"])
        st.markdown("**Loaded Image:**")
        _, center_col, _ = st.columns([1, 1.4, 1])
        with center_col:
            st.image(
                preview_rgb,
                caption=(
                    f"{items[0]['name']} ({preview_rgb.width}×{preview_rgb.height}) | "
                    f"Fruit ID: {fruit_id_from_name(items[0]['name']) or 'N/A'} | "
                    f"Angle: {camera_angle_from_name(items[0]['name'])}"
                ),
                width="stretch",
            )
        return st.button("Run Ripeness Prediction", type="primary", width="stretch")

    st.markdown(f"**Loaded Views ({len(items)} images):**")
    thumb_cols = st.columns(min(len(items), 6))
    for idx, item in enumerate(items):
        with thumb_cols[idx % len(thumb_cols)]:
            st.image(open_rgb_image(item["bytes"]), caption=item["name"], width="stretch")
    return st.button(
        f"Run Ripeness Prediction ({len(items)} Views)",
        type="primary",
        width="stretch",
    )


def render_verdict_banner(title_prefix: str, pred_label: str, confidence: float, extra_info: str) -> None:
    """Render the colored ripeness classification verdict banner."""
    meta = CLASS_METADATA[pred_label]
    st.markdown(
        f"""
        <div class="verdict-card" style="background:{meta['bg_rgba']}; border: 1.5px solid {meta['border_rgba']};">
            <div class="verdict-badge" style="background:{meta['color']}; color:#0F172A;">
                {meta['badge']}
            </div>
            <div class="verdict-title" style="color:{meta['color']};">
                {title_prefix}: {pred_label} ({confidence * 100:.1f}%)
            </div>
            <p class="verdict-desc"><strong>Characteristics:</strong> {meta['description']}</p>
            <p class="verdict-desc"><strong>Recommendation:</strong> {meta['recommendation']}</p>
            <p class="verdict-desc" style="opacity:0.78; font-size:0.86rem; margin-top:0.45rem;">{extra_info}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_chart(probs: Dict[str, float], title: str = "Ripeness Class Probabilities") -> go.Figure:
    """Build horizontal bar chart of probabilities across Unripe, Ripe, and Overripe."""
    df = pd.DataFrame(
        {
            "Ripeness Class": CLASS_ORDER,
            "Probability (%)": [probs[c] * 100.0 for c in CLASS_ORDER],
        }
    )
    fig = go.Figure(
        go.Bar(
            x=df["Probability (%)"],
            y=df["Ripeness Class"],
            orientation="h",
            marker_color=[CLASS_COLOR_MAP[c] for c in df["Ripeness Class"]],
            text=[f"{v:.2f}%" for v in df["Probability (%)"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Probability (%)",
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(CLASS_ORDER))},
        xaxis={"range": [0, 105]},
        height=270,
        margin=dict(l=15, r=20, t=42, b=20),
    )
    return fig


def build_per_view_chart(df: pd.DataFrame) -> go.Figure:
    """Build grouped bar chart comparing class probabilities across uploaded views."""
    melted = df.melt(
        id_vars=["File", "Camera Angle", "Predicted Class"],
        value_vars=["Prob Unripe", "Prob Ripe", "Prob Overripe"],
        var_name="Class",
        value_name="Probability",
    )
    melted["Class"] = melted["Class"].str.replace("Prob ", "", regex=False)
    melted["Probability (%)"] = melted["Probability"] * 100.0
    fig = px.bar(
        melted,
        x="File",
        y="Probability (%)",
        color="Class",
        barmode="group",
        color_discrete_map=CLASS_COLOR_MAP,
        title="Per-View Class Probabilities",
    )
    fig.update_layout(height=320, margin=dict(l=15, r=15, t=42, b=20))
    return fig


def render_class_probability_cards(probs: Dict[str, float], pred_label: str) -> None:
    """Render per-class probability percentage cards highlighting the winning class."""
    st.markdown("##### Class Probability Breakdown")
    for cls in CLASS_ORDER:
        meta = CLASS_METADATA[cls]
        p_val = probs[cls] * 100.0
        is_winner = cls == pred_label
        border = f"2px solid {meta['color']}" if is_winner else "1px solid rgba(148,163,184,0.22)"
        bg = meta["bg_rgba"] if is_winner else "rgba(148,163,184,0.06)"
        tag = " • PREDICTED" if is_winner else ""
        st.markdown(
            f"""
            <div style="padding:0.7rem 0.95rem; border-radius:10px; border:{border}; background:{bg}; margin-bottom:0.55rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:{meta['color']};">{cls}{tag}</span>
                    <span style="font-weight:800; font-size:1.1rem;">{p_val:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_single_result(res: Dict[str, Any]) -> None:
    render_verdict_banner(
        title_prefix="Predicted Ripeness",
        pred_label=res["pred_label"],
        confidence=res["confidence"],
        extra_info=f"Inference Time: <strong>{res['latency_ms']:.1f} ms</strong> | Input: <code>{res['name']}</code>",
    )
    c_chart, c_cards = st.columns([1.45, 1.0], gap="large")
    with c_chart:
        st.plotly_chart(build_probability_chart(res["probs"]), width="stretch")
    with c_cards:
        render_class_probability_cards(res["probs"], res["pred_label"])


def _render_multiview_result(view_results: List[Dict[str, Any]], agg_strategy: str) -> None:
    rows = [
        {
            "File": r["name"],
            "Fruit ID": r["fruit_id"],
            "Camera Angle": r["camera_angle"],
            "Predicted Class": r["pred_label"],
            "Confidence (%)": round(r["confidence"] * 100.0, 2),
            "Prob Unripe": r["probs"]["Unripe"],
            "Prob Ripe": r["probs"]["Ripe"],
            "Prob Overripe": r["probs"]["Overripe"],
            "Latency (ms)": round(r["latency_ms"], 1),
        }
        for r in view_results
    ]
    df = pd.DataFrame(rows)
    fruit_label, fruit_conf, mean_probs, consistency = aggregate_fruit_predictions(df, agg_strategy)
    mean_probs_dict = {cls: float(mean_probs[i]) for i, cls in enumerate(CLASS_ORDER)}

    render_verdict_banner(
        title_prefix="Fruit-Level Ripeness",
        pred_label=fruit_label,
        confidence=fruit_conf,
        extra_info=(
            f"Aggregated over <strong>{len(df)} views</strong> ({agg_strategy}) &nbsp;|&nbsp; "
            f"Angle Consistency: <strong>{consistency:.1f}%</strong>"
        ),
    )

    c_chart, c_cards = st.columns([1.45, 1.0], gap="large")
    with c_chart:
        st.plotly_chart(
            build_probability_chart(mean_probs_dict, title="Aggregated Fruit-Level Probabilities"),
            width="stretch",
        )
    with c_cards:
        render_class_probability_cards(mean_probs_dict, fruit_label)

    st.plotly_chart(build_per_view_chart(df), width="stretch")

    st.markdown("##### Per-View Predictions")
    st.dataframe(
        df.style.format({"Prob Unripe": "{:.4f}", "Prob Ripe": "{:.4f}", "Prob Overripe": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        label="Download Multi-View Predictions CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="durian_multiview_predictions.csv",
        mime="text/csv",
    )


def render_prediction_results(
    prediction_mode: str,
    view_results: List[Dict[str, Any]],
    agg_strategy: str,
) -> None:
    """Render either single-image or aggregated multi-view prediction outputs."""
    if prediction_mode == MODE_SINGLE:
        _render_single_result(view_results[0])
        return
    _render_multiview_result(view_results, agg_strategy)
