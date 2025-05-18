import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Header
st.markdown(
    "<div style='background-color: #0066cc; padding: 20px; border-radius: 10px;'>"
    "<h1 style='text-align: center; color: white; font-size: 30px;'>🌟 Diabetes Prediction App 🌟</h1>"
    "</div>",
    unsafe_allow_html=True,
)

# Tabs for Navigation
tabs = st.tabs(["Home 🏠", "Prediction 🔍", "Insights 📊"])

# Load Dataset
df = pd.read_csv("diabetes.csv")
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)
df[columns_to_replace] = df[columns_to_replace].fillna(df[columns_to_replace].mean())
X = df.drop(columns=['Outcome'])
Y = df['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Home Tab
with tabs[0]:
    st.markdown(
        "Welcome to the 🌟 **Diabetes Prediction App!** 🌟 This app uses machine learning models to predict diabetic conditions "
        "based on user inputs. Explore insights and make predictions on the likelihood of diabetes."
        
        " This app is based on the **PIMA Indian Diabetes Dataset** from **Kaggle**."
    )

    st.markdown("### 📜 Dataset Summary")
    st.write(df.describe()) 

# Prediction Tab
with tabs[1]:
    st.markdown("### 🔍 Make a Prediction")
    st.markdown("Enter values for the following features to predict the likelihood of diabetes:")
    user_input = []
    for column in X.columns:
        value = st.slider(
            f"{column} :", 
            min_value=0.0,
            max_value=float(df[column].max()),
            step=0.1,
            help=f"Adjust the value for {column} ",
        )
        user_input.append(value)

    model_name = st.selectbox("🔢 Select a model:", ["KNN", "Logistic Regression", "Random Forest", "SVC", "Decision Tree"])
    model = None
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)

    elif model_name == "SVC":
        model = SVC()

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        
    elif model_name == "KNN":
        param_grid = {
            'n_neighbors': range(1, 21),
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean', 'manhattan']
        }
        grid_search = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, Y_train)
        model = grid_search.best_estimator_
        st.sidebar.write("Best KNN Parameters:", grid_search.best_params_)

    model.fit(X_train, Y_train)

    if st.button("Predict 🎯"):
        input_data = np.array([user_input]).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = "Diabetes 😷" if prediction[0] == 1 else "No Diabetes 😊"
        st.markdown(
            f"<div style='background-color: #28a745; padding: 20px; border-radius: 10px;'>"
            f"<h2 style='color: white;'> Prediction: {result}</h2>"
            "</div>",
            unsafe_allow_html=True,
        )

# Insights Tab with EDA and Model Evaluation
with tabs[2]:
    st.markdown("### 📊 Dataset Insights and EDA")

    # Correlation Heatmap
    if st.checkbox("📝 Show Correlation Heatmap"):
        st.markdown("#### Heatmap of Feature Correlations 🔥")
        fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)  # Generate heatmap
        st.pyplot(fig)  # Pass the figure explicitly to st.pyplot

    # Pairplot
    if st.checkbox("📉 Show Pairplot"):
        st.markdown("#### Pairplot of Features 🔎")
        pairplot = sns.pairplot(df, hue="Outcome", diag_kind="kde", palette="husl")
        pairplot.fig.set_size_inches(10, 6)  # Set figure size
        st.pyplot(pairplot.fig)  # Use the Pairplot's figure attribute

    # Outcome Distribution
    if st.checkbox("📊 Show Outcome Distribution"):
        st.markdown("#### Distribution of Target Variable 🧮")
        outcome_counts = df['Outcome'].value_counts()  # Count occurrences of each outcome
        fig, ax = plt.subplots()  # Create figure and axes
        outcome_counts.plot(kind='bar', color=['#3498db', '#e74c3c'], ax=ax)  # Generate bar chart
        ax.set_xlabel("Outcome")  # Label the X-axis
        ax.set_ylabel("Count")  # Label the Y-axis
        st.pyplot(fig)  # Pass the figure explicitly to st.pyplot

    # Confusion Matrix
    if st.checkbox("🔎 Show Confusion Matrix"):
        st.markdown("#### Confusion Matrix of Model Performance 🎯")
        try:
            y_pred = model.predict(X_test)  # Predictions on test data
            cm = confusion_matrix(Y_test, y_pred)  # Generate confusion matrix
            fig, ax = plt.subplots(figsize=(4, 3))  # Create figure and axes
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax,
                        xticklabels=['No Diabetes', 'Diabetes'], 
                        yticklabels=['No Diabetes', 'Diabetes'])  # Generate heatmap
            ax.set_xlabel("Predicted labels", fontsize=10)  # Label X-axis
            ax.set_ylabel("True labels", fontsize=10)  # Label Y-axis
            st.pyplot(fig)  # Pass figure explicitly to st.pyplot
        except Exception as e:
            st.error("Please train the model first to view the confusion matrix.")

    # Model Evaluation Metrics
    if st.checkbox("📈 Show Model Evaluation Metrics"):
        st.markdown("#### Detailed Performance Metrics of the Selected Model 🧐")
        try:
            y_pred = model.predict(X_test)  # Predictions on test data
            
            # Compute Evaluation Metrics
            accuracy = accuracy_score(Y_test, y_pred)
            st.write(f"**Accuracy**: {accuracy:.2f}")
            
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
            precision = precision_score(Y_test, y_pred)
            recall = recall_score(Y_test, y_pred)
            f1 = f1_score(Y_test, y_pred)
            
            st.write(f"**Precision**: {precision:.2f}")
            st.write(f"**Recall (Sensitivity)**: {recall:.2f}")
            st.write(f"**F1-Score**: {f1:.2f}")
            
            # ROC Curve and AUC
            if hasattr(model, "predict_proba"):  # Check if the model supports probability prediction
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(Y_test, y_proba)
                st.write(f"**ROC-AUC Score**: {auc:.2f}")
                
                # Plot ROC Curve
                st.markdown("#### ROC Curve 📉")
                fig, ax = plt.subplots()
                RocCurveDisplay.from_predictions(Y_test, y_proba, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("ROC Curve is not available for this model (e.g., SVC without probability support).")
        
        except Exception as e:
            st.error("Please train the model first to view evaluation metrics.")