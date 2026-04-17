import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.trainer import train
from src.predictor import predict


st.title("AutoML Pro")

st.markdown("""
End-to-end machine learning system that automatically:
- Detects problem type
- Preprocesses data
- Compares models
- Selects the best model
- Generates predictions
""")


st.set_page_config(page_title="AutoML Pro", layout="wide")

st.title(" AutoML Pro - End to End ML System")

file = st.file_uploader("Upload your CSV file", type=["csv"])
menu = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "EDA", "Train", "Predict"]
)


if file is not None:

    #  Safety check
    if file.size == 0:
        st.error(" Uploaded file is empty")
        st.stop()

    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Shape: {df.shape}")

    target = st.selectbox(" Select Target Column", df.columns)
    st.write("Target Distribution:")
    st.write(df[target].value_counts())

    if df[target].nunique() < 2:
        st.warning("Target has only one class. Training will fail.")

    if df[target].value_counts().min() < 5:
        st.warning("Dataset is highly imbalanced.") 
    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)

    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.subheader(" Exploratory Data Analysis")

        col1, col2 = st.columns(2)  

    with col1:
        st.write("### Correlation Heatmap")
        try:
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True),annot=True, ax=ax)
            st.pyplot(fig)
        except:
            st.warning("Not enough numeric data for correlation")

    with col2:
        st.write("### Target Distribution")
        try:
            fig2, ax2 = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax2)
            st.pyplot(fig2)
        except:
            st.warning("Cannot plot target distribution")

    st.subheader(" Train Model")

    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            try:
                score, problem_type, leaderboard = train(df, target)

                st.success(" Model Trained Successfully!")
                with open("artifacts/model.pkl", "rb") as f:
                    st.download_button(
                            label="Download Model",
                            data=f,
                            file_name="model.pkl"
                            )
                st.write(f"**Problem Type:** {problem_type}")
                st.write(f"**Best Score:** {score}")

           
                st.subheader(" Model Leaderboard")

                leaderboard_df = pd.DataFrame(leaderboard)
                leaderboard_df = leaderboard_df.sort_values(by="score", ascending=False)

                st.dataframe(    leaderboard_df.style.highlight_max(axis=0),use_container_width=True)

            # Highlight best model
                best_model_name = leaderboard_df.iloc[0]["model"]
                st.success(f" Best Model: {best_model_name}")

            
                st.subheader(" Model Comparison")
                st.caption("Scores are based on cross-validation performance.")
                fig, ax = plt.subplots()
                ax.bar(leaderboard_df["model"], leaderboard_df["score"])
                ax.set_ylabel("Score")
                ax.set_xlabel("Model")
                ax.set_title("Model Performance Comparison")

                st.pyplot(fig)

            except ValueError as ve:
                st.error(f" Data Issue: {ve}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.subheader(" Make Prediction")

    input_data = {}

    for col in df.columns:
        if col != target:
            if df[col].dtype in ["int64", "float64"]:
                input_data[col] = st.number_input(col)
            else:
                input_data[col] = st.text_input(col)

    if st.button("Predict"):
        try:
            # Convert numeric inputs properly
            for key in input_data:
                try:
                    input_data[key] = float(input_data[key])
                except:
                    pass

            result = predict(input_data)

            st.success(f" Prediction: {result}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")