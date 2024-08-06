import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import plotly.express as px
import plotly.graph_objects as go

# Dictionary of model file paths and full names
model_files = {
    "KNN": ("knn_model.joblib", "K-Nearest Neighbors"),
    "Logistic Regression": ("logistic_regression_model.joblib", "Logistic Regression"),
    "Decision Tree": ("decision_tree_model.joblib", "Decision Tree"),
    "Random Forest": ("random_forest_model.joblib", "Random Forest"),
    "GBM": ("gbm_model.joblib", "Gradient Boosting Machine"),
    "XGBM": ("xgbm_model.joblib", "Extreme Gradient Boosting Machine"),
    "Adaboost": ("adaboost_model.joblib", "Adaboost"),
    "Light GBM": ("light_gbm_model.joblib", "Light Gradient Boosting Machine"),
    "CatBoost": ("catboost_model.joblib", "CatBoost"),
    "Naive Bayes": ("naive_bayes_model.joblib", "Naive Bayes"),
    "Voting": ("voting_model.joblib", "Voting Classifier")
}

# Selected features
selected_features = ['protocol_type', 'flag', 'src_bytes', 'dst_bytes', 'count',
                     'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
                     'dst_host_same_srv_rate', 'dst_host_same_src_port_rate']

# Function to load a model
def load_model(model_name):
    return joblib.load(model_files[model_name][0])

# Function to preprocess data
def preprocess_data(data):
    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Handle missing data
    data.fillna(method='ffill', inplace=True)

    # Label encoding for categorical features
    categorical_features = ['protocol_type', 'service', 'flag']
    for feature in categorical_features:
        if feature in data.columns:
            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature])

    # Ensure all selected features are present
    data = data[selected_features]

    # Scaling data
    scaler = StandardScaler()
    data[selected_features] = scaler.fit_transform(data[selected_features])

    return data

# Streamlit interface
st.set_page_config(page_title="AI-Powered Threat Detection", layout="wide", initial_sidebar_state="expanded")
st.title("AI-Powered Threat Detection for Fintech")

# Apply dark mode
st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1avcm0n {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1v3fvcr {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1cpxqw2 {
        background-color: #0e1117;
        color: #ffffff;
    }
    .css-1d391kg {
        background-color: #0e1117;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# File uploader
uploaded_file = st.file_uploader("Upload your test CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Preprocess the data
    data = preprocess_data(data)

    # Dictionary to store model results
    model_results = {}

    # Show loading spinner while making predictions
    with st.spinner('Making predictions...'):
        # Iterate through all models and calculate the percentage of positive predictions
        for model_name, (model_path, full_name) in model_files.items():
            model = load_model(model_name)
            predictions = model.predict(data)
            positive_percentage = (predictions.sum() / len(predictions)) * 100
            model_results[full_name] = positive_percentage

    # Convert results to DataFrame
    results_df = pd.DataFrame(list(model_results.items()), columns=['Model', 'Positive Prediction Percentage'])

    # Apply conditional formatting
    def color_positive_percentage(val):
        if val >= 90:
            color = 'red'
        elif val >= 70:
            color = 'orange'
        elif val >= 60:
            color = 'yellow'
        else:
            color = 'green'
        return f'background-color: {color}'

    # Display the results table with conditional formatting
    st.write("### Model Performance")
    st.dataframe(results_df.style.applymap(color_positive_percentage, subset=['Positive Prediction Percentage']))

    # Identify the best model
    best_model = results_df.loc[results_df['Positive Prediction Percentage'].idxmax()]

    # Define mitigation suggestions
    suggestions = {
        'green': [
            "Maintain regular system updates.",
            "Ensure all software is up-to-date.",
            "Conduct regular security audits."
        ],
        'yellow': [
            "Increase monitoring of network traffic.",
            "Review and update firewall rules.",
            "Conduct a thorough security assessment."
        ],
        'orange': [
            "Implement stricter access controls.",
            "Increase frequency of security training for employees.",
            "Deploy advanced threat detection systems."
        ],
        'red': [
            "Initiate an immediate incident response.",
            "Isolate affected systems to prevent spread.",
            "Engage with cybersecurity experts for a detailed investigation."
        ]
    }

    # Determine threat level and select random suggestion
    if best_model['Positive Prediction Percentage'] >= 90:
        threat_level = 'red'
        threat_description = "Critical threat detected! Immediate action is required."
    elif best_model['Positive Prediction Percentage'] >= 70:
        threat_level = 'orange'
        threat_description = "High threat detected. Prompt action is recommended."
    elif best_model['Positive Prediction Percentage'] >= 60:
        threat_level = 'yellow'
        threat_description = "Moderate threat detected. Increased vigilance is advised."
    else:
        threat_level = 'green'
        threat_description = "Low threat detected. Continue with regular security practices."

    suggestion = random.choice(suggestions[threat_level])

    # Display the best model with descriptive text and suggestion
    st.write("### Best Model")
    # Display the best model details in an expander
    with st.expander("## Show Best Model Details Details"):
        st.write(f"The best model is **{best_model['Model']}** with a positive prediction percentage of **{best_model['Positive Prediction Percentage']:.2f}%**.")
        st.write(f"**Threat Level:** {threat_description}")
        st.write(f"**Suggested Action:** {suggestion}")

    # Create a chart for the table visualizations
    fig = px.bar(results_df, x='Model', y='Positive Prediction Percentage', title="Model Performance")
    st.plotly_chart(fig)

    # Additional visualizations
    st.write("### Additional Model Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(results_df, names='Model', values='Positive Prediction Percentage', title="Model Distribution")
        st.plotly_chart(fig1)
    with col2:
        fig2 = px.scatter(results_df, x='Model', y='Positive Prediction Percentage', title="Model Scatter Plot")
        st.plotly_chart(fig2)

    # Analytics Dashboard
    st.write("### Analytics Dashboard")
    st.write("#### Summary Statistics")
    st.write(data.describe())

    st.write("#### Feature Importance (Random Forest)")
    rf_model = load_model("Random Forest")
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig3 = px.bar(feature_importance, x='Feature', y='Importance', title="Feature Importance")
    st.plotly_chart(fig3)

else:
    st.write("Please upload a CSV file for prediction.")