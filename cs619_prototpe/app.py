import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set Streamlit layout
st.set_page_config(layout="wide")

# Load and preprocess data
st.title("Instructor Dashboard for E-Learning Systems")
st.write("This dashboard provides visualizations and predictions on student assessments using machine learning.")

@st.cache
def load_data():
    # Load the dataset
    student_assessment_df = pd.read_csv('Student_Assessment_Preprocessed.csv')
    return student_assessment_df

data = load_data()
st.write("### Sample of Student Assessment Data")
st.dataframe(data.head())

# Encode target and preprocess data for model
label_encoder = LabelEncoder()
data['Student_final_result'] = label_encoder.fit_transform(data['Student_final_result'])
data = pd.get_dummies(data, columns=['Highest_education', 'Age_band', 'Code_module'])

X = data.drop('Student_final_result', axis=1)
y = data['Student_final_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Heatmap of Correlations
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
st.pyplot(plt)

# Train models and display results
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="linear", random_state=42)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# Display Model Comparison
st.write("### Model Comparison - Accuracy")
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=accuracies, palette="Set2")
plt.title("Model Comparison - Accuracy")
plt.ylabel("Accuracy")
st.pyplot(plt)

# Confusion Matrix for Random Forest
st.write("### Confusion Matrix (Random Forest)")
plt.figure(figsize=(8, 6))
sns.heatmap(results["Random Forest"]['conf_matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt)

# Feature Importance
st.write("### Feature Importance (Random Forest)")
importances = models['Random Forest'].feature_importances_
feature_names = X.columns
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances_df)
plt.title("Feature Importance")
st.pyplot(plt)

# Actual vs Predicted Plot for Random Forest
st.write("### Actual vs Predicted Results (Random Forest)")
best_model = models["Random Forest"]
y_pred_rf = best_model.predict(X_test_scaled)

results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_rf})
results_df = results_df.sort_index()

plt.figure(figsize=(12, 6))
plt.plot(results_df["Actual"].reset_index(drop=True), label="Actual", marker="o", linestyle="-", color="blue")
plt.plot(results_df["Predicted"].reset_index(drop=True), label="Predicted", marker="o", linestyle="-", color="red")
plt.xlabel("Sample Index")
plt.ylabel("Result")
plt.legend()
st.pyplot(plt)
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set Streamlit layout
st.set_page_config(layout="wide")

# Load and preprocess data
st.title("Instructor Dashboard for E-Learning Systems")
st.write("This dashboard provides visualizations and predictions on student assessments using machine learning.")

@st.cache
def load_data():
    # Load the dataset
    student_assessment_df = pd.read_csv('Student_Assessment_Preprocessed.csv')
    return student_assessment_df

data = load_data()
st.write("### Sample of Student Assessment Data")
st.dataframe(data.head())

# Encode target and preprocess data for model
label_encoder = LabelEncoder()
data['Student_final_result'] = label_encoder.fit_transform(data['Student_final_result'])
data = pd.get_dummies(data, columns=['Highest_education', 'Age_band', 'Code_module'])

X = data.drop('Student_final_result', axis=1)
y = data['Student_final_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Heatmap of Correlations
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
st.pyplot(plt)

# Train models and display results
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="linear", random_state=42)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# Display Model Comparison
st.write("### Model Comparison - Accuracy")
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=accuracies, palette="Set2")
plt.title("Model Comparison - Accuracy")
plt.ylabel("Accuracy")
st.pyplot(plt)

# Confusion Matrix for Random Forest
st.write("### Confusion Matrix (Random Forest)")
plt.figure(figsize=(8, 6))
sns.heatmap(results["Random Forest"]['conf_matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt)

# Feature Importance
st.write("### Feature Importance (Random Forest)")
importances = models['Random Forest'].feature_importances_
feature_names = X.columns
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances_df)
plt.title("Feature Importance")
st.pyplot(plt)

# Actual vs Predicted Plot for Random Forest
st.write("### Actual vs Predicted Results (Random Forest)")
best_model = models["Random Forest"]
y_pred_rf = best_model.predict(X_test_scaled)

results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_rf})
results_df = results_df.sort_index()

plt.figure(figsize=(12, 6))
plt.plot(results_df["Actual"].reset_index(drop=True), label="Actual", marker="o", linestyle="-", color="blue")
plt.plot(results_df["Predicted"].reset_index(drop=True), label="Predicted", marker="o", linestyle="-", color="red")
plt.xlabel("Sample Index")
plt.ylabel("Result")
plt.legend()
st.pyplot(plt)
