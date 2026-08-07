"""Streamlit front-end for the Robo Data Scientist AutoML platform."""

from datetime import datetime

import streamlit as st
import pandas as pd

from utils.auto_target_identifier import detect_task_type
from utils.data_utils import (
    analyze_and_prepare_target,
    detect_target_column,
    load_dataset,
)
from utils.model_trainer import PRIMARY_METRIC, train_models
from utils.predict import make_prediction
from utils.model_artifact import ModelArtifact
from utils.constants import MODEL_DIR, ensure_directories
from utils.logging_utils import configure_logging, get_logger

configure_logging()
logger = get_logger("streamlit_app")

st.set_page_config(page_title="Robo Data Scientist 🤖", layout="wide")
st.title("🤖 Robo Data Scientist - AutoML App")
st.markdown("Upload any structured dataset (CSV/Excel), and we'll train 15+ models, rank them, and let you make predictions!")

# Create runtime directories once, at the application entrypoint.
ensure_directories()


# Sidebar for file upload and training
st.sidebar.header("📁 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    df = load_dataset(file)
    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    # Suggest a target column, but let the user override it.
    suggested_target = detect_target_column(df)
    try:
        default_index = list(df.columns).index(suggested_target)
    except ValueError:
        default_index = max(len(df.columns) - 1, 0)


    target_col = st.sidebar.selectbox(
        "🎯 Select Target Column",
        df.columns,
        index=default_index
    )
    st.sidebar.success(f"Selected target: {target_col}")

    if st.sidebar.button("🚀 Train Models"):
        with st.spinner("Training models... This may take a while! ⏳"):
            try:
                # Separate features from the (cleaned/encoded) target.
                X_raw, y, label_encoder = analyze_and_prepare_target(
                    df.copy(), target_col
                )
                task_type = detect_task_type(pd.Series(y))

                st.info(f"Detected task type: **{task_type}**. Training models...")

                # train_models splits BEFORE fitting the preprocessor, so the
                # held-out fold never influences imputation, scaling, outlier
                # detection or feature selection.
                result = train_models(X_raw, pd.Series(y), task_type)

                st.subheader("🏆 Model Leaderboard")
                if not result.leaderboard.empty:
                    st.dataframe(result.leaderboard, use_container_width=True)
                else:
                    st.info("No models were successfully trained or evaluated.")

                # Conditional saving and download button
                if result.best_estimator is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = (
                        MODEL_DIR
                        / f"model_{result.best_model_name}_{timestamp}.pkl"
                    )

                    # Persist a single, self-describing artifact so training and
                    # inference always agree on the serialized contract.
                    ModelArtifact(
                        model=result.best_estimator,
                        preprocessor=result.preprocessor,
                        task_type=task_type,
                        target_column=target_col,
                        label_encoder=label_encoder,
                        model_name=result.best_model_name or "",
                        feature_names=result.feature_names,
                        metrics=result.best_metrics,
                    ).save(str(model_path))

                    metric_name = PRIMARY_METRIC[task_type]
                    best_score = result.best_metrics.get(metric_name)
                    st.success(
                        f"Best model: **{result.best_model_name}** "
                        f"({metric_name}: {best_score:.4f})"
                        if isinstance(best_score, float)
                        else f"Best model: **{result.best_model_name}**"
                    )
                    st.caption(f"Saved to `{model_path}`")

                    st.download_button(
                        label="📥 Download Best Model",
                        data=model_path.read_bytes(),
                        file_name=model_path.name,
                        mime="application/octet-stream",
                    )
                else:
                    st.warning(
                        "No model could be trained successfully. Check the "
                        "leaderboard's Error column and the application logs."
                    )

            except Exception as exc:
                st.error(f"Training failed: {exc}")
                logger.exception("Training failed")

    # Prediction section
    st.sidebar.markdown("---")
    st.sidebar.header("🧪 Make Predictions")
    pred_file = st.sidebar.file_uploader(
        "Upload Data for Prediction", type=["csv", "xlsx"], key="pred_uploader"
    )

    model_files = sorted(
        (p.name for p in MODEL_DIR.glob("*.pkl")), reverse=True
    )
    if not model_files:
        st.sidebar.info("Train a model first to enable predictions.")
    else:
        selected_model = st.sidebar.selectbox(
            "Choose Saved Model", model_files, key="model_selector"
        )

        if pred_file:
            with st.spinner("Making predictions..."):
                try:
                    preds_df = make_prediction(
                        load_dataset(pred_file), str(MODEL_DIR / selected_model)
                    )
                    st.subheader("🔮 Predictions")
                    st.dataframe(preds_df, use_container_width=True)

                    st.download_button(
                        label="📥 Download Predictions",
                        data=preds_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    logger.exception("Prediction failed")
else:
    st.info("👈 Upload a dataset in the sidebar to get started.")

