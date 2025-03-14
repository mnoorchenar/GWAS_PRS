import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ----------------------------
# FUNCTION TO GENERATE SYNTHETIC DATA
# ----------------------------
def generate_synthetic_data(n_samples, snp_effects, demo_effects, noise_level, disease_ratio):
    np.random.seed(42)

    # Generate demographic features
    age = np.random.randint(50, 90, size=n_samples)
    sex = np.random.choice([0, 1], size=n_samples)  # 1 = Male, 0 = Female
    BMI = np.random.uniform(15, 40, size=n_samples)
    PhysicalActivity = np.random.randint(0, 11, size=n_samples)
    ProteinDay = np.random.uniform(50, 150, size=n_samples)
    CurrentSmoking = np.random.choice([0, 1], size=n_samples)
    HighFatMass = np.random.choice([0, 1], size=n_samples)

    # Generate SNPs with specified frequencies
    snp_columns = ["rs10001", "rs10002", "rs10003"]
    snp_data = np.random.choice([0, 1, 2], size=(n_samples, len(snp_columns)), p=[0.6, 0.3, 0.1])

    # Create a DataFrame
    data = pd.DataFrame({
        "age": age,
        "sex": sex,
        "BMI": BMI,
        "PhysicalActivity": PhysicalActivity,
        "ProteinDay": ProteinDay,
        "CurrentSmoking": CurrentSmoking,
        "HighFatMass": HighFatMass
    })

    # Add SNPs to the DataFrame
    snp_df = pd.DataFrame(snp_data, columns=snp_columns)
    data = pd.concat([data, snp_df], axis=1)

    # Calculate the probability of sarcopenia using logistic regression equation
    base_probability = 0.5  # Default base probability

    # Calculate risk score using user-defined effects
    risk_score = (
        snp_effects[0] * data["rs10001"] +
        snp_effects[1] * data["rs10002"] +
        snp_effects[2] * data["rs10003"] +
        demo_effects[0] * data["age"] +
        demo_effects[1] * data["sex"] +
        demo_effects[2] * data["BMI"] +
        demo_effects[3] * data["PhysicalActivity"] +
        demo_effects[4] * data["ProteinDay"] +
        demo_effects[5] * data["CurrentSmoking"] +
        demo_effects[6] * data["HighFatMass"]
    )

    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    risk_score += noise

    # Convert risk score to probability using sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    probability = sigmoid(base_probability + risk_score)

    # Convert probability to binary outcome based on user-defined disease ratio
    disease_threshold = np.percentile(probability, 100 - disease_ratio)
    data["sarcopenia"] = (probability > disease_threshold).astype(int)

    # Calculate PRS as sum of SNP effects
    data["prs"] = (
        snp_effects[0] * data["rs10001"] +
        snp_effects[1] * data["rs10002"] +
        snp_effects[2] * data["rs10003"]
    )

    return data


# Inject custom CSS for global styling and slider layout
st.markdown("""
<style>
/* Target the main app container */
div.stApp {
    font-family: 'Arial', sans-serif;
    margin: 10px;
}

/* Headings within markdown or Streamlit components */
h1, div[data-testid="stMarkdownContainer"] h1 {
    font-size: 1.5em;
    margin-bottom: 0.5em;
}
h2, div[data-testid="stMarkdownContainer"] h2 {
    font-size: 1.3em;
    margin-bottom: 0.25em;
}
h3, div[data-testid="stMarkdownContainer"] h3 {
    font-size: 1.15em;
    margin-bottom: 0.2em;
}
h4, div[data-testid="stMarkdownContainer"] h4 {
    font-size: 1.1em;
    margin-bottom: 0.2em;
}
h5, div[data-testid="stMarkdownContainer"] h5 {
    font-size: 1.1em;
    margin-bottom: 0.2em;
}

p, div[data-testid="stMarkdownContainer"] p {
    font-size: 0.9em;
    margin-bottom: 0.2em;
}

/* Expander styling */
div[data-testid="stExpander"] {
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

/* Slider layout using Flexbox */
.slider-row {
    display: flex;
    align-items: center;
    width: 100%;
    margin-bottom: 4px;
}
.slider-label {
    font-size: 14px !important;
    font-weight: bold;
    width: 160px;
    padding-right: 15px;
    white-space: nowrap;
}
.slider-cell {
    flex-grow: 1;
}
div[data-testid="stSlider"] {
    width: 100% !important;
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SAMPLE GENERATION SECTION
# ----------------------------
st.title("🧬 GWAS Polygenic Risk Score (PRS) Application")

st.markdown("### 🔬 Generate Synthetic SNP & Demographic Data")

# Compact layout for sample size, noise level, and disease percentage
cols = st.columns(3)
n_samples = cols[0].slider("📊 Number of Samples", min_value=100, max_value=2000, value=500, step=50)
noise_level = cols[1].slider("🔊 Noise Level", 0.0, 2.0, 0.5, 0.1)
disease_ratio = cols[2].slider("🩺 % Diseased Patients", 0, 100, 50, 5)

# User-defined effect sizes for SNPs
st.markdown("#### 🧪 SNP Effect Sizes")
col1, col2, col3 = st.columns(3)
snp_effects = [
    col1.slider("rs10001 Effect", -10.0, 10.0, -0.5, 0.1),
    col2.slider("rs10002 Effect", -10.0, 10.0, 0.3, 0.1),
    col3.slider("rs10003 Effect", -10.0, 10.0, 0.7, 0.1)
]

# User-defined effect sizes for demographic features
st.markdown("#### 🏥 Demographic Feature Effect Sizes")
cols = st.columns(4)
demo_effects = [
    cols[0].slider("Age", -10.0, 10.0, -0.5, 0.1),
    cols[1].slider("Sex", -10.0, 10.0, 0.2, 0.1),
    cols[2].slider("BMI", -10.0, 10.0, 0.9, 0.1),
    cols[3].slider("Physical Activity", -10.0, 10.0, 0.5, 0.1),
]
cols = st.columns(3)
demo_effects += [
    cols[0].slider("Protein Intake", -10.0, 10.0, 0.6, 0.1),
    cols[1].slider("Smoking", -10.0, 10.0, -0.8, 0.1),
    cols[2].slider("High Fat Mass", -10.0, 10.0, -0.5, 0.1)
]

# Generate synthetic data
if st.button("🔄 Generate Data"):
    df = generate_synthetic_data(n_samples, snp_effects, demo_effects, noise_level, disease_ratio)
    st.session_state.df_original = df.copy()
    st.session_state.snp_effects = snp_effects  # Store SNP effects for later use
    st.write("### 🔍 Preview of Generated Data:")
    st.dataframe(df.head())

    # Define features and target
    target_variable = "sarcopenia"
    selected_snp_columns = ["rs10001", "rs10002", "rs10003"]
    demographic_features = ["age", "sex", "BMI", "PhysicalActivity", "ProteinDay", "CurrentSmoking", "HighFatMass"]
    model_features = selected_snp_columns + demographic_features
    st.session_state.model_features = model_features

    # Train/Test split
    X = df[model_features]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Train a default model
    default_model = DecisionTreeClassifier(max_depth=3)
    default_model.fit(X_train, y_train)
    st.session_state.model = default_model

# ----------------------------
# PATIENT SELECTION AND DISPLAY
# ----------------------------
if "df_original" in st.session_state:
    df = st.session_state.df_original
    selected_index = st.selectbox("🧑‍⚕️ Select a Patient", df.index, key="patient_select")
    st.session_state.selected_index = selected_index

    # Display patient-specific results
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.write("### 🏥 Patient Demographics")
        original_values = df.iloc[selected_index:selected_index + 1]
        sex_icon = "♂️ Male" if original_values["sex"].values[0] == 1 else "♀️ Female"
        st.markdown(f"""
        - 📅 **Age:** {original_values["age"].values[0]}
        - {sex_icon}
        - ⚖️ **BMI:** {original_values["BMI"].values[0]:.1f}
        - 🚶 **Physical Activity:** {original_values["PhysicalActivity"].values[0]}
        - 🍗 **Protein Intake:** {original_values["ProteinDay"].values[0]:.1f} g
        - 🚬 **Current Smoker:** {"✅ Yes" if original_values["CurrentSmoking"].values[0] == 1 else "❌ No"}
        - 🍔 **High Fat Mass:** {"✅ Yes" if original_values["HighFatMass"].values[0] == 1 else "❌ No"}
        """)

    with col2:
        st.write("### 🔬 SNP Effect Sizes")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(["rs10001", "rs10002", "rs10003"], st.session_state.snp_effects, color="blue")
        ax.set_xlabel("Effect Size (β)")
        ax.set_ylabel("SNPs")
        ax.set_title("SNP Effect Sizes")
        st.pyplot(fig)

    # Display the Polygenic Risk Score Box
    polygenic_score = df.loc[selected_index, "prs"]
    box_color = "rgba(255, 0, 0, 0.2)" if polygenic_score > 0 else "rgba(0, 255, 0, 0.2)"
    st.markdown(f"""
    <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px;">
         🧬 Polygenic Risk Score (PRS): <strong>{polygenic_score:.3f}</strong>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# TRAINING MODEL SECTION
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

        # Compact slider function
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

        if st.button("🚀 Train Model") and "X_train" in st.session_state:
            model_params = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "random_state": 42
            }
            if n_estimators is not None:
                model_params["n_estimators"] = n_estimators

            model = model_options[selected_model](**model_params)
            model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model = model
            st.write(f"### ✅ Model Trained with {selected_model}")

    with col_output:
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

# ----------------------------
# SHAP WATERFALL PLOT
# ----------------------------
if "model" in st.session_state and "X" in st.session_state and "selected_index" in st.session_state:
    st.write("### 🔍 SHAP Waterfall Plot for Selected Patient")
    instance = st.session_state.X.iloc[[st.session_state.selected_index]]

    explainer = shap.TreeExplainer(st.session_state.model)
    shap_values = explainer(instance)

    if len(shap_values.values.shape) > 1:
        class_index = 1
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

    plt.figure(figsize=(3, 2))
    shap.plots.waterfall(shap_values_single, show=False)
    fig_waterfall = plt.gcf()
    st.pyplot(fig_waterfall)
    plt.close(fig_waterfall)

    # Model prediction probability
    pred_prob = st.session_state.model.predict_proba(instance)[0, 1]

    # Explanation
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    baseline_prob = 0.50
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

    shap_vals = np.array(shap_values_single.values)
    abs_shap_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap_vals)[-4:][::-1]
    top_features = [shap_values_single.feature_names[i] for i in top_indices]
    top_values = [shap_values_single.values[i] for i in top_indices]

    for feat, val in zip(top_features, top_values):
        delta_prob = sigmoid(val) - 0.50
        if val > 0:
            direction = "increases"
            color = "#FF0000"
        else:
            direction = "decreases"
            color = "#0000FF"
        st.markdown(
            f"- <b>{feat}</b>: contributes {val:+.2f} in log-odds, roughly a {delta_prob:+.2%} change in probability. "
            f"(i.e. it <span style='color:{color};'>{direction}</span> the baseline probability of 50%).",
            unsafe_allow_html=True
        )
