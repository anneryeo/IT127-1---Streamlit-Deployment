import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, classification_report, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# Run (as regular Python script): 
# & "D:/Developer/Projects/Academic Projects/IT127-1 - Streamlit Deployment/.venv/Scripts/python.exe" "D:/Developer/Projects/Academic Projects/IT127-1 - Streamlit Deployment/app.py"

# Run (as Streamlit web app): 
# & "D:/Developer/Projects/Academic Projects/IT127-1 - Streamlit Deployment/.venv/Scripts/python.exe" -m streamlit run "D:/Developer/Projects/Academic Projects/IT127-1 - Streamlit Deployment/app.py"

#Dataset source: https://archive.ics.uci.edu/dataset/73/mushroom

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🔬")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🔬")

    def load_data():
        try:
            data = pd.read_csv("mushrooms.csv")
        except FileNotFoundError:
            st.error("mushrooms.csv file not found. Please upload the dataset.")
            st.stop()
        
        original_target = data['type'].copy()
        
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        
        st.write(f"Dataset shape: {data.shape}")
        st.write(f"Number of unique classes in target: {len(data['type'].unique())}")
        st.write(f"Original target values: {original_target.unique()}")
        st.write(f"Encoded target values: {data['type'].unique()}")
        
        return data

    @st.cache_data
    def split(df):
        y = df.type
        X = df.drop(columns=["type"])
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            y_pred = model.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Create confusion matrix heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            unique_classes = sorted(list(set(y_test) | set(y_pred)))
            class_labels = [f"Class {i}" for i in unique_classes]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_labels, yticklabels=class_labels, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            plt.close()
        
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            # Check if this is a binary classification problem
            if len(set(y_test)) == 2:
                y_prob = model.predict_proba(x_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC curve plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random classifier')
                ax.set_xlim((0.0, 1.0))
                ax.set_ylim((0.0, 1.05))
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                st.write("ROC Curve is only available for binary classification")
        
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            # Check if this is a binary classification problem
            if len(set(y_test)) == 2:
                y_prob = model.predict_proba(x_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                
                # Create Precision-Recall curve plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recall, precision, color='blue', lw=2, 
                       label='Precision-Recall curve')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend(loc="lower left")
                ax.grid(True, alpha=0.3)
                ax.set_xlim((0.0, 1.0))
                ax.set_ylim((0.0, 1.05))
                st.pyplot(fig)
                plt.close()
            else:
                st.write("Precision-Recall Curve is only available for binary classification")

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    
    # Get the actual unique classes in the target variable
    unique_classes = sorted(df['type'].unique())
    class_names = [f"Class {i}" for i in unique_classes]
    
    st.write(f"Working with {len(unique_classes)} classes: {unique_classes}")

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", float(np.round(precision_score(y_test, y_pred, average='weighted'), 2)))
            st.write("Recall: ", float(np.round(recall_score(y_test, y_pred, average='weighted'), 2)))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", float(np.round(precision_score(y_test, y_pred, average='weighted'), 2)))
            st.write("Recall: ", float(np.round(recall_score(y_test, y_pred, average='weighted'), 2)))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            bootstrap_bool = True if bootstrap == 'True' else False
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap_bool, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", float(np.round(precision_score(y_test, y_pred, average='weighted'), 2)))
            st.write("Recall: ", float(np.round(recall_score(y_test, y_pred, average='weighted'), 2)))
            plot_metrics(metrics, model, x_test, y_test, class_names)

if __name__ == "__main__":
    main()