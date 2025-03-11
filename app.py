import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Inject custom CSS for global styling and slider layout
st.markdown("""
    <style>
    /* Global body styling */
    body {
        font-family: 'Arial', sans-serif;
        margin: 10px;
    }
    /* Heading styles */
    h1 { font-size: 2em; margin-bottom: 0.5em; }
    h2 { font-size: 1.75em; margin-bottom: 0.25em; }
    h3 { font-size: 1.5em; margin-bottom: 0.15em; }
    h4 { font-size: 1.25em; margin-bottom: 0.15em; }
    h5 { font-size: 1em; margin-bottom: 0.15em; }
    h6 { font-size: 1em; margin-bottom: 0.5em; }
    p { margin: 0.5em 0; }

    /* Expander Styling */
    .st-expander {
        background-color: #d9d9d9 !important;
        color: black !important;
        border-radius: 8px;
        padding: 10px;
    }
    .expander-content {
        max-height: 70vh;  
        overflow-y: auto;  
        background-color: #f5f5f5 !important;
        padding: 12px;
        border-radius: 10px;
    }

    /* Table-like layout for slider rows */
    .slider-row {
        display: table;
        width: 100%;
        margin-bottom: 4px;
    }
    .slider-label {
        display: table-cell;
        font-size: 14px !important;
        font-weight: bold;
        width: 160px;  /* Adjust width as needed */
        padding-right: 15px;  /* Padding between label and slider */
        white-space: nowrap;
    }
    .slider-cell {
        display: table-cell;
        width: 100%;
    }
    /* Ensure slider widget takes full width of its container */
    div[data-testid="stSlider"] {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# FILE UPLOAD SECTION
# ----------------------------
st.title("🧬 GWAS Polygenic Risk Score (PRS) Application")

uploaded_file = st.file_uploader("📂 Upload your SNP dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()  # Remove any missing values from the entire dataset

    if "df_original" not in st.session_state:
        st.session_state.df_original = df.copy()

    st.write("### 🔍 Preview of Uploaded Data:")
    st.dataframe(df.head())

    target_variable = st.selectbox("🎯 Select Target Variable", df.columns)
    st.markdown("<br>", unsafe_allow_html=True)
    selected_snp_columns = st.multiselect("🧪 Select SNPs", df.columns)
    demographic_features = st.multiselect("🏥 Include Demographic Features", df.columns)

    if selected_snp_columns and target_variable:
        # Create model features and store them in session state
        model_features = selected_snp_columns + demographic_features
        st.session_state.model_features = model_features

        # beta_values = np.random.uniform(-1, 1, len(selected_snp_columns))

        # Calculate beta values via logistic regression for each SNP
        beta_values = []
        for snp in selected_snp_columns:
            # Build a temporary feature matrix including the SNP and all demographic covariates
            # If no demographic features are selected, we use only the SNP.
            if demographic_features:
                X_temp = df[[snp] + demographic_features]
            else:
                X_temp = df[[snp]]

            # Initialize and fit logistic regression model
            lr_model = LogisticRegression(max_iter=1000)
            lr_model.fit(X_temp, df[target_variable])

            # Extract the coefficient for the SNP. Here, we assume the SNP is the first column.
            beta = lr_model.coef_[0][0]
            beta_values.append(beta)

        beta_values = np.array(beta_values)

        df["PRS"] = np.dot(df[selected_snp_columns], beta_values)

        X = df[model_features]
        y = df[target_variable]

        # Store dataset and splits in session state
        st.session_state.X = X
        st.session_state.y = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        # Train a default model for preview (optional)
        default_model = DecisionTreeClassifier(max_depth=3)
        default_model.fit(X_train, y_train)
        st.session_state.model = default_model

        # Select Patient for Display
        selected_index = st.selectbox("🧑‍⚕️ Select a Patient", df.index)
        st.session_state.selected_index = selected_index

        # ----------------------------
        # DISPLAY PATIENT RESULTS
        # ----------------------------
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.write("### 🏥 Patient Demographics")
            original_values = st.session_state.df_original.iloc[selected_index:selected_index + 1]
            sex_icon = "♂️ Male" if original_values["sex"].values[0] == 1 else "♀️ Female"
            st.markdown(f"""
            - 📅 **Age:** {original_values["age"].values[0]}
            - {sex_icon}
            - ⚖️ **BMI:** {original_values["BMI"].values[0]}
            - 🚶 **Physical Activity:** {original_values["Physicalactivity"].values[0]}
            - 🍗 **Protein Intake:** {original_values["ProteinDay"].values[0]}
            - 🚬 **Current Smoker:** {"✅ Yes" if original_values["CurrentSmoking"].values[0] == 1 else "❌ No"}
            """)
        with col2:
            st.write("### 🔬 SNP Effect Sizes")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(selected_snp_columns, beta_values, color="blue")
            ax.set_xlabel("Effect Size (β)")
            ax.set_ylabel("SNPs")
            ax.set_title("SNP Effect Sizes")
            st.pyplot(fig)

# Display the Polygenic Risk Score Box
        polygenic_score = df.loc[selected_index, 'PRS']
        box_color = "rgba(255, 0, 0, 0.2)" if polygenic_score > 0 else "rgba(0, 255, 0, 0.2)"
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px;">
             🧬 Polygenic Risk Score (PRS) : <strong>{polygenic_score:.3f}</strong>
        </div>
        """, unsafe_allow_html=True)
# ----------------------------
# TRAINING MODEL SECTION (Labels & Sliders with Fixed Padding)
# This section only runs after the user clicks "Train Model"
# ----------------------------
st.write("## 🔬 Train Your Prediction Model")

with st.expander("🔽 Click to expand/collapse model settings & training results", expanded=False):
    st.markdown('<div class="expander-content">', unsafe_allow_html=True)
    col_hyper, col_output = st.columns([2, 3])
    with col_hyper:
        st.markdown("<h4>⚙️ Model Settings</h4>", unsafe_allow_html=True)
        model_options = {
            "🌲 Random Forest": RandomForestClassifier,
            "🔥 Gradient Boosting": GradientBoostingClassifier,
            "🧩 Decision Tree": DecisionTreeClassifier
        }
        selected_model = st.selectbox("🔍 Select Model", list(model_options.keys()))

        # Compact slider function that provides a non-empty label
        def compact_slider(icon, label_text, min_val, max_val, default_val, key):
            non_empty_label = label_text if label_text.strip() != "" else "slider"
            col1, col2 = st.columns([1.5, 5])
            with col1:
                st.markdown(f'<div class="slider-row"><span class="slider-label">{icon} {label_text}:</span></div>', unsafe_allow_html=True)
            with col2:
                return st.slider(non_empty_label, min_value=min_val, max_value=max_val, value=default_val, key=key, label_visibility="collapsed")

        if selected_model != "🧩 Decision Tree":
            n_estimators = compact_slider("🌲", "Trees", 10, 500, 100, "trees_slider")
        else:
            n_estimators = None
        max_depth = compact_slider("✏️", "Depth", 2, 20, 5, "depth_slider")
        min_samples_split = compact_slider("✂️", "Split Min", 2, 20, 2, "split_slider")
        min_samples_leaf = compact_slider("🍃", "Leaf Min", 1, 10, 1, "leaf_slider")

        if st.button("🚀 Train Model"):
            model_params = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "random_state": 42
            }
            if n_estimators is not None:
                model_params["n_estimators"] = n_estimators

            # Train the model using the stored training data
            model = model_options[selected_model](**model_params)
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model = model
            st.write(f"### ✅ Model Trained with {selected_model}")

    with col_output:
        # Only show training results if a model has been trained
        if "model" in st.session_state:
            model = st.session_state.model
            accuracy = model.score(st.session_state.X_test, st.session_state.y_test)
            st.markdown(f"<h4>📊 Accuracy: {accuracy:.2%}</h4>", unsafe_allow_html=True)
            st.markdown("<h5>📈 Feature Importance</h5>", unsafe_allow_html=True)
            fig_fi, ax_fi = plt.subplots(figsize=(4, 3))
            ax_fi.barh(st.session_state.model_features, model.feature_importances_, color="orange")
            ax_fi.set_xlabel("Importance Score")
            ax_fi.set_title("Feature Importance")
            st.pyplot(fig_fi)
        else:
            st.write("Train your model to see the results.")
    st.markdown('</div>', unsafe_allow_html=True)

if "model" in st.session_state and "X" in st.session_state and "selected_index" in st.session_state:
    st.write("### 🔍 SHAP Waterfall Plot for Selected Patient")
    instance = st.session_state.X.iloc[[st.session_state.selected_index]]

    explainer = shap.TreeExplainer(st.session_state.model)
    shap_values = explainer(instance)

    if len(shap_values.values.shape) > 1:
        class_index = 1
        # Here we force a baseline of 1 in log-odds space. (Note: log(odds) for 50% is 0, but you can adjust if needed)
        shap_values_single = shap.Explanation(
            values=shap_values.values[..., class_index].flatten(),
            base_values=1,
            data=shap_values.data.flatten(),
            feature_names=shap_values.feature_names
        )
    else:
        shap_values_single = shap.Explanation(
            values=shap_values.values.flatten(),
            base_values=shap_values.base_values.flatten(),
            data=shap_values.data.flatten(),
            feature_names=shap_values.feature_names
        )

    # Explicitly create a new figure to prevent overlapping plots
    plt.figure()
    shap.plots.waterfall(shap_values_single, show=False)
    fig_waterfall = plt.gcf()
    st.pyplot(fig_waterfall)
    plt.close(fig_waterfall)

    # Get the model prediction probability (assumes predict_proba returns probability for class 1)
    pred_prob = st.session_state.model.predict_proba(instance)[0, 1]


    # --- Explanation: Convert log-odds contributions to probability change ---
    # Define the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # For explanation, we assume a baseline probability of 50%
    baseline_prob = 0.50
    # Note: In a truly calibrated logistic model, the log-odds baseline would be 0 (since sigmoid(0)=0.50),
    # but here the waterfall's base value is set to 1 as per your code.
    # You can adjust this baseline if needed.

    st.markdown("### Explanation of Top 4 Important Features (in Probability Terms)")
    st.markdown(
        f"<b>Model Prediction:</b> {pred_prob:.2f} (i.e. {pred_prob * 100:.0f}%)<br>"
        f"<b>Baseline (assumed):</b> 0.50 (50%)<br>"
        f"Thus, the features collectively shift the prediction by {pred_prob - baseline_prob:+.2f} "
        f"(i.e. {(abs(pred_prob - baseline_prob)) * 100:.0f} percentage points).",
        unsafe_allow_html=True
    )
    st.markdown("""
    <ul>
      <li><span style="color:#FF0000;"><b>Red features</b></span> increase the probability from 50%.</li>
      <li><span style="color:#0000FF;"><b>Blue features</b></span> decrease the probability from 50%.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Compute top 4 features based on absolute SHAP values (in log-odds)
    shap_vals = np.array(shap_values_single.values)
    abs_shap_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap_vals)[-4:][::-1]
    top_features = [shap_values_single.feature_names[i] for i in top_indices]
    top_values = [shap_values_single.values[i] for i in top_indices]

    # For each top feature, convert its log-odds contribution to a probability change
    for feat, val in zip(top_features, top_values):
        # Calculate the probability if only this feature's log-odds were added to the baseline log-odds (assumed 0 for 50%)
        # Here, we use: new_prob = sigmoid(val) and then change = new_prob - 0.50.
        delta_prob = sigmoid(val) - 0.50
        if val > 0:
            direction = "increases"
            color = "#FF0000"  # red
        else:
            direction = "decreases"
            color = "#0000FF"  # blue
        st.markdown(
            f"- <b>{feat}</b>: contributes {val:+.2f} in log-odds, which roughly translates to a change of {delta_prob:+.2%} in probability. "
            f"(i.e. it <span style='color:{color};'>{direction}</span> the baseline probability of 50%).",
            unsafe_allow_html=True
        )
