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
# ----------------------------
# FUNCTION TO GENERATE SYNTHETIC DATA
# ----------------------------
def generate_synthetic_data(n_samples, snp_effects, demo_effects, noise_level, disease_ratio):
    # Generate demographic features
    age = np.random.randint(50, 90, size=n_samples)
    sex = np.random.choice([0, 1], size=n_samples)
    BMI = np.random.uniform(15, 40, size=n_samples)
    PhysicalActivity = np.random.randint(0, 11, size=n_samples)
    ProteinDay = np.random.uniform(50, 150, size=n_samples)
    CurrentSmoking = np.random.choice([0, 1], size=n_samples)
    HighFatMass = np.random.choice([0, 1], size=n_samples)

    # Generate SNPs
    snp_columns = ["rs10001", "rs10002", "rs10003"]
    snp_data = np.random.choice([0, 1, 2], size=(n_samples, len(snp_columns)), p=[0.7, 0.25, 0.05])

    # Create DataFrame
    data = pd.DataFrame({
        "age": age, "sex": sex, "BMI": BMI, "PhysicalActivity": PhysicalActivity,
        "ProteinDay": ProteinDay, "CurrentSmoking": CurrentSmoking, "HighFatMass": HighFatMass
    })
    snp_df = pd.DataFrame(snp_data, columns=snp_columns)
    data = pd.concat([data, snp_df], axis=1)

    # Adjust SNP effects based on noise level
    if noise_level > 1:
        snp_effects = np.random.normal(snp_effects, 0.1 * noise_level, size=len(snp_effects))

    # Base risk score
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

    # Add complexity dynamically based on noise level
    if noise_level > 1:
        risk_score += demo_effects[0] * np.log1p(data["age"])
        risk_score += demo_effects[2] * np.sqrt(data["BMI"])

    if noise_level > 2:
        interaction_terms = (
            0.3 * data["rs10001"] * data["BMI"] +
            0.2 * data["rs10002"] * data["age"] +
            0.5 * data["rs10003"] * data["CurrentSmoking"]
        )
        risk_score += interaction_terms

    if noise_level > 3:
        risk_score += demo_effects[3] * (data["PhysicalActivity"] ** 2)

    # Add noise scaled to the complexity
    noise = np.random.normal(0, noise_level * np.std(risk_score), n_samples)
    risk_score += noise

    # Convert log-odds to probability
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    probability = sigmoid(0.3 + risk_score)

    # Store continuous risk score
    data["risk_score"] = probability

    # Adjust disease classification threshold dynamically
    disease_threshold = np.percentile(probability, 100 - disease_ratio)

    # Introduce randomness in classification for high-noise scenarios
    if noise_level > 2:
        random_shift = np.random.normal(0, 0.05 * noise_level, size=n_samples)
    else:
        random_shift = 0

    data["disease"] = (probability + random_shift > disease_threshold).astype(int)

    # Ensure at least two classes exist
    if len(data["disease"].unique()) == 1:
        flip_indices = np.random.choice(data.index, size=max(1, int(0.05 * n_samples)), replace=False)
        data.loc[flip_indices, "disease"] = 1 - data.loc[flip_indices, "disease"]

    return data

# Inject custom CSS
st.markdown("""
<style>
div.stApp {
    font-family: 'Arial', sans-serif;
    margin: 10px;
}
h1, div[data-testid="stMarkdownContainer"] h1 { font-size: 1.5em; margin-bottom: 0.5em; }
h2, div[data-testid="stMarkdownContainer"] h2 { font-size: 1.3em; margin-bottom: 0.25em; }
h3, div[data-testid="stMarkdownContainer"] h3 { font-size: 1.15em; margin-bottom: 0.2em; }
h4, div[data-testid="stMarkdownContainer"] h4 { font-size: 1.1em; margin-bottom: 0.2em; }
h5, div[data-testid="stMarkdownContainer"] h5 { font-size: 1.1em; margin-bottom: 0.2em; }
p, div[data-testid="stMarkdownContainer"] p { font-size: 0.9em; margin-bottom: 0.2em; }
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
.slider-cell { flex-grow: 1; }
div[data-testid="stSlider"] { width: 100% !important; max-width: 100%; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🧬 GWAS Polygenic Risk Score (PRS) Application")
st.markdown("### 🔬 Generate Synthetic SNP & Demographic Data")

cols = st.columns(3)
n_samples = cols[0].slider("📊 Number of Samples", min_value=100, max_value=1000, value=100, step=50)
noise_level = cols[1].slider(
    "🔊 Noise Level (Data Complexity)",
    0.0, 4.0, 1.0, 0.5,
    format="%.1f"
)

# Map numerical values to descriptive categories
if noise_level < 1:
    noise_description = "🟢 Low (Mostly Linear Features)"
elif noise_level < 2:
    noise_description = "🟡 Medium (Some Non-Linear Effects)"
elif noise_level < 3:
    noise_description = "🟠 High (Feature Interactions & Complexity)"
else:
    noise_description = "🔴 Very High (Strong Noise, Quadratic Effects)"

# Centering description inside column [1] using HTML & CSS
st.markdown(
    f"""
    <div style="text-align: center; font-size: 12px; font-weight: bold;">
        {noise_description}
    </div>
    """,
    unsafe_allow_html=True
)

disease_ratio = cols[2].slider("🩺 % Diseased Patients", 0, 100, 50, 5)

st.markdown("#### 🧪 SNP Effect Sizes")
col1, col2, col3 = st.columns(3)
snp_effects = [
    col1.slider("rs10001 Effect", -2.0, 2.0, -0.5, 0.1),
    col2.slider("rs10002 Effect", -2.0, 2.0, 0.3, 0.1),
    col3.slider("rs10003 Effect", -2.0, 2.0, 0.7, 0.1)
]

st.markdown("#### 🏥 Demographic Feature Effect Sizes")
cols = st.columns(4)
demo_effects = [
    cols[0].slider("Age", -2.0, 2.0, -0.5, 0.1),
    cols[1].slider("Sex", -2.0, 2.0, 0.2, 0.1),
    cols[2].slider("BMI", -2.0, 2.0, 0.9, 0.1),
    cols[3].slider("Physical Activity", -2.0, 2.0, 0.5, 0.1),
]
cols = st.columns(3)
demo_effects += [
    cols[0].slider("Protein Intake", -2.0, 2.0, 0.6, 0.1),
    cols[1].slider("Smoking", -2.0, 2.0, -0.8, 0.1),
    cols[2].slider("High Fat Mass", -2.0, 2.0, -0.5, 0.1)
]

if st.button("🔄 Generate Data"):
    df = generate_synthetic_data(n_samples, snp_effects, demo_effects, noise_level, disease_ratio)

    # Train Logistic Regression to Estimate SNP Effect Sizes
    logistic_model = LogisticRegression(max_iter=500)
    new_snp_effects = {}

    for snp in ["rs10001", "rs10002", "rs10003"]:
        X_logit = df[["age", "sex", "BMI", "PhysicalActivity", "ProteinDay", "CurrentSmoking", "HighFatMass", snp]]
        y_logit = df["disease"]

        # ✅ Ensure at least two classes exist in target variable before training
        if len(y_logit.unique()) > 1:
            logistic_model.fit(X_logit, y_logit)
            new_snp_effects[snp] = logistic_model.coef_[0][-1]
        else:
            new_snp_effects[snp] = 0  # If only one class exists, avoid training error

    # Compute PRS using new effect sizes
    df["prs"] = (
        new_snp_effects["rs10001"] * df["rs10001"] +
        new_snp_effects["rs10002"] * df["rs10002"] +
        new_snp_effects["rs10003"] * df["rs10003"]
    )

    # ✅ Ensure session state variables exist before use
    if "df_original" not in st.session_state:
        st.session_state.df_original = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None


    st.session_state.df_original = df.copy()
    st.write("### 🔍 Preview of Generated Data:")
    st.dataframe(df.head())

    # Store dataset for modeling
    target_variable = "disease"
    model_features = ["age", "sex", "BMI", "PhysicalActivity", "ProteinDay", "CurrentSmoking", "HighFatMass", "rs10001",
                      "rs10002", "rs10003"]

    X = df[model_features]
    y = df[target_variable]

    # ✅ Ensure dataset is not empty before splitting
    if X.empty or y.empty:
        st.error("🚨 Error: The generated dataset is empty. Adjust effect sizes or noise levels.")
    else:
        # ✅ Split data & store in session state
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.model_features = model_features
        st.session_state.snp_effects = new_snp_effects

        st.write("### ✅ Data Generated Successfully! You can now train your model.")

    # ✅ Train Decision Tree Classifier only if training data exists
    if st.session_state.X_train is not None and st.session_state.y_train is not None:
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model

        # ✅ Check model accuracy only if training completed
        accuracy = model.score(st.session_state.X_test, st.session_state.y_test)
        st.write(f"### ✅ Model Accuracy: {accuracy:.2%}")
    else:
        st.error("🚨 Error: No training data available. Generate data first.")

# ----------------------------
# PATIENT SELECTION AND DISPLAY
# ----------------------------
if "df_original" in st.session_state and st.session_state.df_original is not None:
    df = st.session_state.df_original
    selected_index = st.selectbox("🧑‍⚕️ Select a Patient", df.index, key="patient_select")
    st.session_state.selected_index = selected_index

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

        # ✅ Ensure SNP effects exist before plotting
        if "snp_effects" in st.session_state and st.session_state.snp_effects:
            snp_fig = plt.figure(figsize=(4, 3))
            ax = snp_fig.add_subplot(111)
            ax.barh(["rs10001", "rs10002", "rs10003"], list(st.session_state.snp_effects.values()), color="blue")
            ax.set_xlabel("Effect Size (β)")
            ax.set_ylabel("SNPs")
            ax.set_title("SNP Effect Sizes")
            st.pyplot(snp_fig)
            plt.close(snp_fig)
        else:
            st.write("SNP effect sizes are not available yet. Please generate data first.")

    # ✅ Ensure PRS is available before displaying
    if "prs" in df.columns:
        polygenic_score = df.loc[selected_index, "prs"]
        box_color = "rgba(255, 0, 0, 0.2)" if polygenic_score > 0 else "rgba(0, 255, 0, 0.2)"
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px;">
             🧬 Polygenic Risk Score (PRS): <strong>{polygenic_score:.3f}</strong>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# TRAINING MODEL SECTION (Labels & Sliders with Fixed Padding)
# This section only runs after the user clicks "Train Model"
# ----------------------------
st.write("## 🔬 Train Your Prediction Model")

# ----------------------------
# TRAINING MODEL SECTION
# ----------------------------
with st.expander("🔽 Click to expand/collapse model settings & training results", expanded=True):
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

        # Compact slider function for hyperparameters
        def compact_slider(icon, label_text, min_val, max_val, default_val, key):
            col1, col2 = st.columns([1.5, 5])
            with col1:
                st.markdown(f'<div class="slider-row"><span class="slider-label">{icon} {label_text}:</span></div>', unsafe_allow_html=True)
            with col2:
                return st.slider(label_text, min_value=min_val, max_value=max_val, value=default_val, key=key, label_visibility="collapsed")

        if selected_model != "🧩 Decision Tree":
            n_estimators = compact_slider("🌲", "Trees", 10, 500, 100, "trees_slider")
        else:
            n_estimators = None
        max_depth = compact_slider("✏️", "Depth", 2, 20, 5, "depth_slider")
        min_samples_split = compact_slider("✂️", "Split Min", 2, 20, 2, "split_slider")
        min_samples_leaf = compact_slider("🍃", "Leaf Min", 1, 10, 1, "leaf_slider")

        if st.button("🚀 Train Model"):
            # ✅ Ensure that training data exists before proceeding
            if "X_train" not in st.session_state or "y_train" not in st.session_state:
                st.error("🚨 Error: No training data found. Please generate data first.")
            else:
                model_params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                }
                if n_estimators is not None:
                    model_params["n_estimators"] = n_estimators

                # Train the selected model
                model = model_options[selected_model](**model_params)
                model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.model = model
                st.write(f"### ✅ Model Trained with {selected_model}")

    with col_output:
        if "model" in st.session_state:
            model = st.session_state.model
            accuracy = model.score(st.session_state.X_test, st.session_state.y_test)
            st.markdown(f"<h4>📊 Accuracy: {accuracy:.2%}</h4>", unsafe_allow_html=True)

            # Feature importance visualization
            if hasattr(model, "feature_importances_"):
                st.markdown("<h5>📈 Feature Importance</h5>", unsafe_allow_html=True)
                # Sorting features by importance in descending order
                sorted_indices = np.argsort(model.feature_importances_)#[::-1]  # Sorting from max to min
                sorted_features = np.array(st.session_state.model_features)[sorted_indices]
                sorted_importance = model.feature_importances_[sorted_indices]
                fig_fi, ax_fi = plt.subplots(figsize=(3, 2))
                ax_fi.barh(sorted_features, sorted_importance, color="orange")
                ax_fi.set_xlabel("Importance Score")
                ax_fi.set_title("Feature Importance")
                st.pyplot(fig_fi)
        else:
            st.write("Train your model to see the results.")

    st.markdown('</div>', unsafe_allow_html=True)


if "model" in st.session_state and "X" in st.session_state and "selected_index" in st.session_state:
    st.write("### 📈📉 SHAP Waterfall Plot for Selected Patient")
    st.write("🔍 The **graph shows log-odds (logits), which is how the model calculates the risk.** However, what really matters is the final **predicted probability of disease**.")

    instance = st.session_state.X.iloc[[st.session_state.selected_index]]

    # Compute SHAP values
    explainer = shap.TreeExplainer(st.session_state.model)
    shap_values = explainer(instance)

    # Handle classification models with multiple classes
    if len(shap_values.values.shape) > 1:
        class_index = 1 if shap_values.values.shape[-1] > 1 else 0
        base_values = float(np.ravel(shap_values.base_values[..., class_index])[0])  # Extract scalar
        shap_values_single = shap.Explanation(
            values=shap_values.values[..., class_index].flatten(),
            base_values=base_values,
            data=shap_values.data.flatten(),
            feature_names=shap_values.feature_names
        )
    else:
        base_values = float(np.ravel(shap_values.base_values)[0])  # Extract scalar
        shap_values_single = shap.Explanation(
            values=shap_values.values.flatten(),
            base_values=base_values,
            data=shap_values.data.flatten(),
            feature_names=shap_values.feature_names
        )

    # Convert log-odds to probability
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    baseline_prob = sigmoid(base_values)  # Convert base logit to probability
    final_prob = sigmoid(base_values + sum(shap_values_single.values))  # Final predicted prob

    # Compute top 3 important features based on absolute SHAP values
    shap_vals = np.array(shap_values_single.values)
    abs_shap_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap_vals)[-3:][::-1]
    top_features = [shap_values_single.feature_names[i] for i in top_indices]
    top_values = [shap_values_single.values[i] for i in top_indices]

    # Convert feature contributions from log-odds to probability shifts
    top_explanations = []
    for feat, val in zip(top_features, top_values):
        prob_change = float(sigmoid(base_values + val) - baseline_prob)  # Convert to float
        color = "#FF0000" if prob_change > 0 else "#0000FF"
        direction = "increases" if prob_change > 0 else "decreases"
        top_explanations.append(
            f"<span style='color:{color};'><b>{feat}</b> changes probability by {prob_change:+.2%}.</span>"
        )

    # Display results in two columns: 2/3 for the plot, 1/3 for explanation
    col1, col2 = st.columns([4, 2])

    with col1:
        # Explicitly create and close the SHAP waterfall plot
        plt.figure()
        shap.plots.waterfall(shap_values_single, show=False)
        fig_waterfall = plt.gcf()
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

    with col2:
        st.markdown("### Explanation")
        st.markdown(
            f"This patient's estimated risk is **{final_prob:.1%}**, starting from an average of **{baseline_prob:.1%}**."
            "<br> 🔴 **Red** features increase the probability, while 🔵 **blue** features decrease it."
            "<br> Most influential factors: " + ", ".join(top_explanations) + ".",
            unsafe_allow_html=True
        )


#
# if "model" in st.session_state and "X" in st.session_state and "selected_index" in st.session_state:
#     st.write("### 🔍 SHAP Waterfall Plot for Selected Patient")
#
#     instance = st.session_state.X.iloc[[st.session_state.selected_index]]
#
#     # Compute SHAP values
#     explainer = shap.TreeExplainer(st.session_state.model)
#     shap_values = explainer(instance)
#
#     # Handle classification models with multiple classes
#     if len(shap_values.values.shape) > 1:
#         class_index = 1 if shap_values.values.shape[-1] > 1 else 0
#         shap_values_single = shap.Explanation(
#             values=shap_values.values[..., class_index].flatten(),
#             base_values=shap_values.base_values[..., class_index].flatten(),
#             data=shap_values.data.flatten(),
#             feature_names=shap_values.feature_names
#         )
#     else:
#         shap_values_single = shap.Explanation(
#             values=shap_values.values.flatten(),
#             base_values=shap_values.base_values.flatten(),
#             data=shap_values.data.flatten(),
#             feature_names=shap_values.feature_names
#         )
#
#     # Compute top 3 important features based on absolute SHAP values
#     shap_vals = np.array(shap_values_single.values)
#     abs_shap_vals = np.abs(shap_vals)
#     top_indices = np.argsort(abs_shap_vals)[-3:][::-1]
#     top_features = [shap_values_single.feature_names[i] for i in top_indices]
#     top_values = [shap_values_single.values[i] for i in top_indices]
#
#
#     # Convert log-odds contributions to probability change
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#
#     top_explanations = []
#     for feat, val in zip(top_features, top_values):
#         delta_prob = sigmoid(val) - 0.50
#         color = "#FF0000" if val > 0 else "#0000FF"
#         direction = "increases" if val > 0 else "decreases"
#         top_explanations.append(
#             f"<span style='color:{color};'><b>{feat}</b> ({val:+.2f}) {direction} the prediction.</span>"
#         )
#
#     # Display results in two columns: 2/3 for the plot, 1/3 for explanation
#     col1, col2 = st.columns([2, 1])
#
#     with col1:
#         # Explicitly create and close the SHAP waterfall plot
#         plt.figure()
#         shap.plots.waterfall(shap_values_single, show=False)
#         fig_waterfall = plt.gcf()
#         st.pyplot(fig_waterfall)
#         plt.close(fig_waterfall)
#
#     with col2:
#         st.markdown("### Explanation")
#         st.markdown(
#             "🔴 **Red** features increase the prediction probability, while 🔵 **blue** features decrease it."
#             "<br><br> Most influential factors: " + ", ".join(top_explanations) + ".",
#             unsafe_allow_html=True
#         )
