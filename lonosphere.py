import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load the dataset
df = pd.read_csv("Ionosphere (1).csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Ionosphere Data")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.write("Classification Report:")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Summary plot
st.subheader("Summary Plot")

fig, ax = plt.subplots()
# Check the structure of shap_values
st.write("SHAP Values Structure:", len(shap_values), "classes detected.")
if isinstance(shap_values, list):
    # Print shapes of each element in shap_values
    for i, values in enumerate(shap_values):
        st.write(f"SHAP values for class {i}: {values.shape}")

    if len(shap_values) > 1:
        try:
            shap.summary_plot(shap_values[1], X_test, show=False)
            st_shap(fig)
        except AssertionError:
            st.write("Mismatch in SHAP values shape and feature matrix.")
    else:
        st.write("SHAP values are not available for class 1.")
else:
    st.write("SHAP values do not have the expected list structure.")

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter value for {feature}:", value=float(X_test[feature].mean()), step=0.1)

# Create a DataFrame from input data
input_df = pd.DataFrame(input_data, index=[0])

# Make prediction
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]  # Probability of being in class 1 (presence)

# Display prediction
st.write(f"**Prediction:** {'Presence of Object (Class 1)' if prediction == 1 else 'Empty Air (Class 0)'}")
st.write(f"**Presence Probability:** {probability:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)

# Force plot
st.subheader("Force Plot")
if len(shap_values) > 1:
    try:
        # Check if shap_values_input has the correct index
        st.write("SHAP values input shape:", shap_values_input[1].shape)
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values_input[1], input_df))
    except AssertionError:
        st.write("Mismatch in SHAP values shape for force plot.")
    except IndexError:
        st.write("IndexError: The requested class index is out of bounds.")
else:
    st.write("SHAP force plot for class 1 is not available.")

# Decision plot
st.subheader("Decision Plot")
if len(shap_values) > 1:
    try:
        # Check if shap_values_input has the correct index
        st.write("SHAP values input shape:", shap_values_input[1].shape)
        st_shap(shap.decision_plot(explainer.expected_value[1], shap_values_input[1], X.columns))
    except AssertionError:
        st.write("Mismatch in SHAP values shape for decision plot.")
    except IndexError:
        st.write("IndexError: The requested class index is out of bounds.")
else:
    st.write("SHAP decision plot for class 1 is not available.")
