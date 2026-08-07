# app.py

import os
import base64
from datetime import datetime

import streamlit as st
import pandas as pd

from utils.auto_target_identifier import detect_task_type
from utils.data_utils import (
    analyze_and_prepare_target,
    detect_target_column,
    load_dataset,
)
from utils.model_trainer import train_models
from utils.predict import make_prediction
from utils.model_artifact import ModelArtifact
from utils.constants import MODEL_DIR, ENCODER_DIR
from utils.logging_utils import setup_logger

logger = setup_logger("StreamlitApp")

st.set_page_config(page_title="Robo Data Scientist 🤖", layout="wide")
st.title("🤖 Robo Data Scientist - AutoML App")
st.markdown("Upload any structured dataset (CSV/Excel), and we'll train 15+ models, rank them, and let you make predictions!")

# Ensure MODEL_DIR and ENCODER_DIR exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)


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
                    model_filename = f"model_{result.best_model_name}_{timestamp}.pkl"
                    model_path = os.path.join(MODEL_DIR, model_filename)

                    # Persist a single, self-describing artifact so training and
                    # inference always agree on the serialized contract.
                    ModelArtifact(
                        model=result.best_estimator,
                        preprocessor=result.preprocessor,
                        task_type=task_type,
                        target_column=target_col,
                        label_encoder=label_encoder,
                        model_name=result.best_model_name,
                        feature_names=result.feature_names,
                        metrics=result.best_metrics,
                    ).save(model_path)

                    st.success(
                        f"Training complete! Best model "
                        f"'{result.best_model_name}' saved to '{model_path}'"
                    )

                    with open(model_path, "rb") as file_to_download:
                        btn = st.download_button(
                            label="📥 Download Best Model",
                            data=file_to_download,
                            file_name=os.path.basename(model_path),
                            mime="application/octet-stream"
                        )
                else:
                    st.warning("No best model could be trained successfully. Please check logs for errors or try a different dataset.")


            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.error(f"Training Error: {e}", exc_info=True)
                st.exception(e) # Display full traceback in Streamlit

    # Prediction section
    st.sidebar.markdown("---")
    st.sidebar.header("🧪 Make Predictions")
    pred_file = st.sidebar.file_uploader("Upload Data for Prediction", type=["csv", "xlsx"], key="pred_uploader")

    # List available models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    selected_model = st.sidebar.selectbox("Choose Saved Model", model_files if model_files else ["No model found"], key="model_selector")

    if pred_file and selected_model != "No model found":
        input_df_for_prediction = load_dataset(pred_file)
        model_path_for_prediction = os.path.join(MODEL_DIR, selected_model)

        with st.spinner("Making predictions..."):
            try:
                # Call the make_prediction function from utils.predict
                preds_df = make_prediction(input_df_for_prediction, model_path_for_prediction)
                st.subheader("🔮 Predictions")
                st.dataframe(preds_df)

                # Download predictions
                csv = preds_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.error(f"Prediction Error: {e}", exc_info=True)
                st.exception(e) # Display full traceback in Streamlit
else:
    st.warning("Please upload a dataset to get started.")

