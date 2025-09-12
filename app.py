import streamlit as st
import pandas as pd
import numpy as np 
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

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown

if __name__ == "__main__":
    main()