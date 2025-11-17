import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent", 
    layout="wide",
    page_icon="📊"
)

st.title("📊 Enhanced AI Data Analysis Agent")

# Add warning about Ollama on Render
st.warning("""
⚠️ **Note**: Ollama integration requires local installation and won't work on Render's cloud environment. 
The data cleaning and visualization features will work perfectly!
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    # Load Data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df
        st.success("✅ File uploaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Use session state data
df = st.session_state.df

if df is not None:
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "🧹 Data Cleaning", "📊 Visualization", "🤖 AI Analysis"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data types information
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
        dtype_df.columns = ['Column', 'Data Type']
        st.dataframe(dtype_df, use_container_width=True)
        
        # Missing values breakdown
        st.subheader("Missing Values Breakdown")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualize missing values
            fig_missing = px.bar(missing_df, x='Column', y='Missing Count',
                               title='Missing Values by Column',
                               color='Missing Count')
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 No missing values found!")
    
    with tab2:
        st.header("Data Cleaning Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Handle Missing Values")
            
            # Select columns for missing value treatment
            columns_with_missing = df.columns[df.isnull().any()].tolist()
            
            if columns_with_missing:
                selected_columns = st.multiselect(
                    "Select columns to handle missing values:",
                    columns_with_missing,
                    default=columns_with_missing
                )
                
                treatment_method = st.selectbox(
                    "Treatment method:",
                    ["Remove rows", "Mean imputation", "Median imputation", 
                     "Mode imputation", "Forward fill", "Backward fill", "Custom value"]
                )
                
                if treatment_method == "Custom value":
                    custom_value = st.text_input("Enter custom value:")
                
                if st.button("Apply Missing Value Treatment"):
                    df_clean = df.copy()
                    
                    for col in selected_columns:
                        if treatment_method == "Remove rows":
                            df_clean = df_clean.dropna(subset=[col])
                        elif treatment_method == "Mean imputation":
                            if pd.api.types.is_numeric_dtype(df_clean[col]):
                                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                            else:
                                st.warning(f"Cannot use mean imputation on non-numeric column: {col}")
                        elif treatment_method == "Median imputation":
                            if pd.api.types.is_numeric_dtype(df_clean[col]):
                                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                            else:
                                st.warning(f"Cannot use median imputation on non-numeric column: {col}")
                        elif treatment_method == "Mode imputation":
                            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown", inplace=True)
                        elif treatment_method == "Forward fill":
                            df_clean[col].fillna(method='ffill', inplace=True)
                        elif treatment_method == "Backward fill":
                            df_clean[col].fillna(method='bfill', inplace=True)
                        elif treatment_method == "Custom value" and custom_value:
                            try:
                                # Try to convert to numeric first
                                df_clean[col].fillna(float(custom_value), inplace=True)
                            except:
                                df_clean[col].fillna(custom_value, inplace=True)
                    
                    st.session_state.df = df_clean
                    st.success("Missing value treatment applied!")
                    st.rerun()
            else:
                st.info("No columns with missing values found.")
        
        with col2:
            st.subheader("Handle Duplicates")
            
            duplicate_count = df.duplicated().sum()
            st.write(f"Number of duplicate rows: **{duplicate_count}**")
            
            if duplicate_count > 0:
                if st.button("Remove All Duplicates"):
                    initial_rows = df.shape[0]
                    df_clean = df.drop_duplicates()
                    final_rows = df_clean.shape[0]
                    st.session_state.df = df_clean
                    st.success(f"Removed {initial_rows - final_rows} duplicate rows!")
                    st.rerun()
                
                if st.button("Show Duplicates"):
                    duplicates = df[df.duplicated(keep=False)]
                    st.write("Duplicate rows:")
                    st.dataframe(duplicates, use_container_width=True)
            
            st.subheader("Column Operations")
            selected_column = st.selectbox("Select column for operation:", df.columns)
            
            if st.button("Show Unique Values"):
                unique_vals = df[selected_column].unique()
                st.write(f"Unique values in {selected_column}:")
                st.write(unique_vals)
            
            if st.button("Show Value Counts"):
                value_counts = df[selected_column].value_counts()
                st.write(f"Value counts for {selected_column}:")
                st.dataframe(value_counts, use_container_width=True)
    
    with tab3:
        st.header("Data Visualization")
        
        # Select columns for visualization
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Numeric Data Visualization")
            
            if numeric_columns:
                num_col = st.selectbox("Select numeric column:", numeric_columns)
                
                plot_type = st.selectbox("Select plot type:", 
                                       ["Histogram", "Box Plot", "Density Plot"])
                
                if plot_type == "Histogram":
                    fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Box Plot":
                    fig = px.box(df, y=num_col, title=f"Box Plot of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Density Plot":
                    fig = px.histogram(df, x=num_col, marginal="rug", 
                                     title=f"Density Plot of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found for visualization.")
        
        with col2:
            st.subheader("Categorical Data Visualization")
            
            if categorical_columns:
                cat_col = st.selectbox("Select categorical column:", categorical_columns)
                
                plot_type = st.selectbox("Select categorical plot type:", 
                                       ["Bar Chart", "Pie Chart"])
                
                value_counts = df[cat_col].value_counts().reset_index()
                value_counts.columns = ['Category', 'Count']
                
                if plot_type == "Bar Chart":
                    fig = px.bar(value_counts, x='Category', y='Count', 
                               title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Pie Chart":
                    fig = px.pie(value_counts, names='Category', values='Count',
                               title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns found for visualization.")
        
        # Correlation heatmap for numeric columns
        if len(numeric_columns) > 1:
            st.subheader("Correlation Heatmap")
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix, title="Correlation Heatmap", 
                          aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("AI-Powered Analysis")
        
        st.info("""
        🤖 This section would integrate with AI models like Ollama for advanced analysis.
        Currently showing data insights and statistical analysis.
        """)
        
        # Show current data state
        st.subheader("Current Data State")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head(), use_container_width=True)
        
        # Statistical Analysis
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Data Quality Report
        st.subheader("Data Quality Report")
        quality_report = {
            'Total Rows': df.shape[0],
            'Total Columns': df.shape[1],
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum(),
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
        }
        st.json(quality_report)

    # Download cleaned data in sidebar
    st.sidebar.header("Export Data")
    
    # Convert dataframe to CSV for download
    csv = st.session_state.df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="📥 Download Cleaned Data as CSV",
        data=csv,
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Reset data button
    if st.sidebar.button("🔄 Reset to Original Data"):
        # Reload original data from uploaded file
        uploaded_file.seek(0)  # Reset file pointer
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.rerun()

else:
    st.info("👆 Please upload a CSV or Excel file to get started!")
    
    # Example data
    if st.button("Load Example Data"):
        # Create sample data
        np.random.seed(42)
        example_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', None],
            'Age': [25, 30, 35, None, 28, 32, 29, 31],
            'Salary': [50000, 60000, None, 55000, 52000, 58000, 54000, 56000],
            'Department': ['HR', 'IT', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'IT'],
            'Experience': [2, 5, 8, 3, 2, 6, 4, 5]
        }
        st.session_state.df = pd.DataFrame(example_data)
        st.rerun()