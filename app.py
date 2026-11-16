"""Streamlit Dashboard entrypoint for Durian Ripeness Classification (Vision-Only Student).

Run from repository root:
    python -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from visualization import (
    DEFAULT_BEST_CKPT,
    aggregate_fruit_predictions,
    collect_input_items,
    compute_fingerprint,
    inject_custom_css,
    load_eval_transform,
    load_student_model,
    predict_single_image,
    render_header,
    render_mode_selector,
    render_prediction_results,
    render_preview_and_trigger,
    render_sidebar_controls,
    render_sidebar_model_summary,
    run_views_inference,
)

__all__ = [
    "DEFAULT_BEST_CKPT",
    "aggregate_fruit_predictions",
    "load_eval_transform",
    "load_student_model",
    "predict_single_image",
    "main",
]


def main() -> None:
    """Orchestrate the Vision-Only Student Durian Ripeness Streamlit application."""
    st.set_page_config(
        page_title="Durian Ripeness Classifier | Vision-Only Student",
        page_icon="🍈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    resolved_ckpt_path, device_str = render_sidebar_controls()
    render_header()

    if resolved_ckpt_path is None:
        st.warning("Please select or upload a valid Student checkpoint (.pth) in the sidebar.")
        return

    try:
        model, ckpt_meta = load_student_model(resolved_ckpt_path, device_str)
        eval_transform = load_eval_transform()
    except Exception as exc:
        st.error(f"Failed to load Student model: {exc}")
        return

    render_sidebar_model_summary(ckpt_meta, device_str)

    prediction_mode, agg_strategy = render_mode_selector()
    items = collect_input_items(prediction_mode)

    if not items:
        st.info("Upload or capture durian RGB image(s) above, then click **Run Ripeness Prediction**.")
        return

    run_clicked = render_preview_and_trigger(prediction_mode, items)
    current_fp = compute_fingerprint(prediction_mode, resolved_ckpt_path, device_str, items)

    if run_clicked:
        with st.spinner("Running get_eval_transforms() + VisionStudent inference..."):
            view_results = run_views_inference(model, eval_transform, items, device_str)
            st.session_state["last_prediction"] = {
                "fingerprint": current_fp,
                "mode": prediction_mode,
                "results": view_results,
            }

    saved = st.session_state.get("last_prediction")
    if not saved or saved.get("fingerprint") != current_fp:
        return

    render_prediction_results(prediction_mode, saved["results"], agg_strategy)


if __name__ == "__main__":
    main()
