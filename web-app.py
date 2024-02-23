"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
import streamlit as st


@st.cache()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('training-data.csv')

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
