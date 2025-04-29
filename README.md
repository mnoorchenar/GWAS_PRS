# GWAS Polygenic Risk Score (PRS) Application 🧬

Welcome to the **GWAS PRS Application** — an interactive platform to simulate genetic and demographic data, build predictive models, and explore disease risk factors.

This app is designed for **students**, **researchers**, and **enthusiasts** who want to understand how genetic variations and lifestyle factors contribute to disease risk — **without needing any coding skills**.

👉 [Launch the App](https://gwas-prs.streamlit.app/)

---

## Features 🚀

- **Generate Synthetic Data**
  - Create customized datasets with Single Nucleotide Polymorphisms (SNPs) and demographic features like age, BMI, smoking habits, and more.
  - Adjust the number of samples, disease prevalence, and data complexity (noise level).

- **Customize Effect Sizes**
  - Fine-tune the influence of each SNP and demographic factor on disease outcomes.
  - Simulate simple linear or complex non-linear relationships.

- **Train Machine Learning Models**
  - Choose from Random Forest, Gradient Boosting, or Decision Tree models.
  - Easily tweak model parameters (tree depth, number of trees, etc.) and view training results.

- **Analyze Individual Patients**
  - Select a patient and review their demographics, genetic markers, and predicted risk.
  - Visualize **Polygenic Risk Scores (PRS)** and **feature importance** using SHAP plots.

---

## How to Use 🛠️

### 1. Generate Synthetic Data
Customize your dataset by adjusting:
- Number of samples
- Noise level (data complexity)
- % of diseased patients
- Effect sizes for SNPs and lifestyle factors

![20845D08-266E-4683-965C-580C87A9BF85](https://github.com/user-attachments/assets/95c8f271-605b-4c55-9d9c-e6e3beaa0df6)

---

### 2. Explore Patient Profiles
Pick a patient from the dataset and explore:
- Personal demographics
- SNP effect sizes
- Polygenic Risk Score (PRS)

Understand the patient's risk through intuitive visual explanations and SHAP waterfall plots.

![9882FE40-E483-47F9-851D-6B8D657FE9A9](https://github.com/user-attachments/assets/0148c9e0-2fe8-40f2-816f-3746e9664af4)

---

### 3. Train a Prediction Model
Select a machine learning model and tune its settings:
- Random Forest
- Gradient Boosting
- Decision Tree

After training, you'll immediately see:
- Model accuracy
- Feature importance scores

![1F765BBD-14EB-42F3-8BC9-C12E9E19BB0B](https://github.com/user-attachments/assets/756e3cd8-7721-4258-b1f0-5190a42d2cec)

---

### 4. Interpret Risk with SHAP Waterfall Plot 📈📉

After selecting a patient, the app provides a **SHAP Waterfall Plot** to explain the model's risk prediction in an intuitive way.

🔍 **What it shows:**
- The model calculates risk using **log-odds (logits)**, but what matters most is the final **predicted probability of disease**.
- The plot visualizes how each feature **pushes** the risk **higher (red)** or **lower (blue)** compared to the baseline.

🧪 **Example Interpretation:**
- This patient's estimated risk is **66.1%**, starting from an average population risk of **67.8%**.
- 🔴 Features in **red** (like "Current Smoking") increase the risk.
- 🔵 Features in **blue** (like "Age" and "BMI") decrease the risk.
- **Most influential factors**:
  - `age` changes probability by **-0.60%**
  - `BMI` changes probability by **-0.40%**
  - `CurrentSmoking` changes probability by **-0.35%**

This makes it easy to understand which factors are driving each individual's risk prediction.

![1CBA4639-6EFF-47F2-A22E-1D2BE6937CC0](https://github.com/user-attachments/assets/328dabda-6fa0-4b14-b41f-4a70cc8c2f18)

## Why Use This App? 🎯

- Learn the basics of **genetic risk modeling**.
- Experiment with **data complexity** and **model performance**.
- Get **hands-on experience** in understanding feature importance and explainable AI.

Whether you're prototyping ideas, teaching concepts, or simply exploring how genetics and lifestyle interplay, this app offers a practical and visual approach.

---

## Technologies Used 🛠

- **Python 3**
- **Streamlit** (for the interactive web app)
- **scikit-learn** (for machine learning models)
- **SHAP** (for explainable AI visualizations)
- **Matplotlib** (for charts and plots)

---

## Project Status 📈

The app is fully functional and available for public use. Future updates may include:
- More SNPs and demographic variables
- Advanced modeling options
- Exporting datasets and models
