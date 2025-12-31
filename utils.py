import pandas as pd
from datetime import datetime
import streamlit as st


def load_data(uploaded_file):
    """Loads data from a CSV or Excel file and applies necessary fixes."""
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Fix for PyArrow serialization issues
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].apply(lambda x: isinstance(x, (datetime, pd.Timestamp))).any():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
