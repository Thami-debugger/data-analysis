import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import json
import requests
from datetime import datetime


def generate_rule_based_insights(df):
    """Heuristic data summary, chart suggestion, and data quality fixes without external APIs."""
    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    dup_total = int(df.duplicated().sum())

    top_missing = (
        df.isna().sum().sort_values(ascending=False).head(3)
    )
    top_missing_cols = [f"{col} ({int(cnt)} missing)" for col, cnt in top_missing.items() if cnt > 0]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    summary_parts = [
        f"You have a {rows}-row, {cols}-column dataset.",
        f"Missing values total: {missing_total}.",
        f"Duplicate rows: {dup_total}.",
    ]
    if top_missing_cols:
        summary_parts.append("Top missing: " + ", ".join(top_missing_cols) + ".")
    if numeric_cols:
        summary_parts.append(f"Numeric columns: {', '.join(numeric_cols[:5])}" + ("..." if len(numeric_cols) > 5 else ""))
    if categorical_cols:
        summary_parts.append(f"Categorical columns: {', '.join(categorical_cols[:5])}" + ("..." if len(categorical_cols) > 5 else ""))

    # Chart suggestion
    chart = {"type": "none", "x": None, "y": None, "hue": None, "title": "" , "reasoning": ""}
    if numeric_cols and categorical_cols:
        chart = {
            "type": "bar",
            "x": categorical_cols[0],
            "y": None,
            "hue": None,
            "title": f"Distribution of {categorical_cols[0]}",
            "reasoning": "Categorical distribution shows balance across groups.",
        }
    elif len(numeric_cols) >= 2:
        chart = {
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "hue": None,
            "title": f"Scatter of {numeric_cols[0]} vs {numeric_cols[1]}",
            "reasoning": "Two numeric features—scatter can show correlation or clusters.",
        }
    elif numeric_cols:
        chart = {
            "type": "histogram",
            "x": numeric_cols[0],
            "y": None,
            "hue": None,
            "title": f"Distribution of {numeric_cols[0]}",
            "reasoning": "Single numeric feature—histogram shows skew and spread.",
        }

    fixes = []
    if missing_total:
        fixes.append("Handle missing values: impute numeric with median/mean, categorical with mode or 'Unknown'.")
    if dup_total:
        fixes.append("Remove duplicate rows if they are unintended duplicates.")
    if numeric_cols:
        skewed = df[numeric_cols].skew().abs().sort_values(ascending=False)
        skewed_cols = [col for col, val in skewed.items() if val > 1]
        if skewed_cols:
            fixes.append("Consider log/Box-Cox transform for skewed numeric columns: " + ", ".join(skewed_cols[:4]))
    if categorical_cols:
        high_card = [col for col in categorical_cols if df[col].nunique() > rows * 0.5]
        if high_card:
            fixes.append("High-cardinality categoricals: consider encoding or pruning for " + ", ".join(high_card[:3]))

    narrative = " ".join(summary_parts)
    if missing_total:
        narrative += " The data has gaps; consider a gentle cleanup before modeling."
    if dup_total:
        narrative += " There are repeated rows that could distort aggregates."
    if numeric_cols:
        narrative += " Numeric fields give us room for distributions and correlation checks."
    if categorical_cols:
        narrative += " Categorical columns can reveal balance or imbalance across groups."

    return {
        "summary": narrative,
        "chart": chart,
        "data_fixes": fixes,
    }


def generate_cleaning_suggestions(df):
    """Suggest cleaning actions based on simple data heuristics."""
    suggestions = []

    missing_total = int(df.isna().sum().sum())
    if missing_total:
        suggestions.append(f"Handle missing values ({missing_total} total) via removal or imputation in 'Handle Missing Values'.")

    dup_total = int(df.duplicated().sum())
    if dup_total:
        suggestions.append(f"Remove {dup_total} duplicate rows with 'Remove All Duplicates'.")

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        needs_trim = False
        for col in object_cols:
            col_series = df[col].dropna().astype(str)
            if not col_series.empty and col_series.str.contains(r"^\s|\s$", regex=True).any():
                needs_trim = True
                break
        if needs_trim:
            suggestions.append("Trim whitespace in text columns using 'Trim whitespace in text columns'.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        skewed = df[numeric_cols].skew().abs().sort_values(ascending=False)
        skewed_cols = [col for col, val in skewed.items() if val > 1]
        if skewed_cols:
            suggestions.append("Treat skew/outliers in numeric columns (try 'Remove outliers (IQR)' or transforms) for: " + ", ".join(skewed_cols[:4]))

        zero_var = [col for col in numeric_cols if df[col].nunique() <= 1]
        if zero_var:
            suggestions.append("Drop constant numeric columns that add no signal: " + ", ".join(zero_var[:5]))

    high_card = [col for col in object_cols if df[col].nunique() > len(df) * 0.5]
    if high_card:
        suggestions.append("High-cardinality categoricals detected; consider pruning or encoding: " + ", ".join(high_card[:3]))

    return suggestions


def build_cleaning_plan(df):
    """Construct an auto-clean plan based on heuristics; limited to safe operations."""
    plan = {
        "trim_whitespace": False,
        "drop_duplicates": False,
        "fill_missing": False,
        "drop_constant": [],
    }

    if int(df.duplicated().sum()) > 0:
        plan["drop_duplicates"] = True

    # Trim if any leading/trailing whitespace exists
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        col_series = df[col].dropna().astype(str)
        if not col_series.empty and col_series.str.contains(r"^\s|\s$", regex=True).any():
            plan["trim_whitespace"] = True
            break

    if int(df.isna().sum().sum()) > 0:
        plan["fill_missing"] = True

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plan["drop_constant"] = [col for col in numeric_cols if df[col].nunique() <= 1]

    return plan


def apply_cleaning_plan(df, plan):
    df_clean = df.copy()

    if plan.get("trim_whitespace"):
        object_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        df_clean[object_cols] = df_clean[object_cols].apply(lambda col: col.str.strip())

    if plan.get("drop_duplicates"):
        df_clean = df_clean.drop_duplicates()

    if plan.get("fill_missing"):
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        for col in num_cols:
            if df_clean[col].isna().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        for col in cat_cols:
            if df_clean[col].isna().any():
                mode_val = df_clean[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                df_clean[col].fillna(fill_val, inplace=True)

    drop_constant = plan.get("drop_constant") or []
    if drop_constant:
        df_clean = df_clean.drop(columns=drop_constant)

    return df_clean


def generate_chart_suggestions(df):
    """Produce up to three simple chart configs based on available columns."""
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if numeric_cols:
        suggestions.append({
            "type": "histogram",
            "x": numeric_cols[0],
            "y": None,
            "hue": None,
            "title": f"Distribution of {numeric_cols[0]}",
            "reasoning": "Quick look at skew and spread for a numeric column."
        })

    if len(numeric_cols) >= 2:
        suggestions.append({
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "hue": None,
            "title": f"Scatter of {numeric_cols[0]} vs {numeric_cols[1]}",
            "reasoning": "Check correlation or clustering between two numeric features."
        })

    if categorical_cols:
        suggestions.append({
            "type": "bar",
            "x": categorical_cols[0],
            "y": None,
            "hue": None,
            "title": f"Distribution of {categorical_cols[0]}",
            "reasoning": "Category balance and potential class imbalance."
        })

    return suggestions[:3]


def top_correlations(df, limit=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr().abs()
    corr.values[[np.arange(corr.shape[0])]*2] = 0  # zero self-corr
    pairs = []
    for i, col_i in enumerate(numeric_cols):
        for j, col_j in enumerate(numeric_cols):
            if j <= i:
                continue
            pairs.append(((col_i, col_j), corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:limit]


def outlier_counts(df, iqr_multiplier=1.5, limit=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        if count:
            results.append((col, count))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def build_data_brief(df, max_columns=15, sample_values=3):
    """Prepare a compact, JSON-safe snapshot of the dataframe for local Llama."""
    brief = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [],
        "missing": int(df.isna().sum().sum()),
    }
    for col in df.columns[:max_columns]:
        series = df[col]
        brief["columns"].append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "unique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "examples": [str(v)[:50] for v in series.dropna().unique()[:sample_values]],
            }
        )
    return brief


def call_local_llama(prompt: str, model: str = "llama3.1"):
    """Call Ollama locally; return text or None on failure."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise data storyteller."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content")
    except Exception:
        return None


def detect_semantic_types(df):
    semantics = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            semantics[col] = "boolean"
            continue
        if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) == len(series.dropna()):
            semantics[col] = "id"
            continue
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().astype(str).head(50)
            if sample.str.fullmatch(r"[0-9a-fA-F-]{8,}").mean() > 0.6:
                semantics[col] = "id"
                continue
            if sample.str.contains(r"^\$?\s*-?\d+(,\d{3})*(\.\d+)?$", regex=True).mean() > 0.6:
                semantics[col] = "currency_str"
                continue
            if sample.str.contains(r"^-?\d+(\.\d+)?%$", regex=True).mean() > 0.6:
                semantics[col] = "percent_str"
                continue
        if pd.api.types.is_datetime64_any_dtype(series):
            semantics[col] = "datetime"
            continue
    return semantics


def time_gaps(df, datetime_col):
    try:
        ts = pd.to_datetime(df[datetime_col].dropna()).sort_values()
        if ts.empty:
            return None
        diffs = ts.diff().dropna()
        if diffs.empty:
            return None
        most_common = diffs.mode().iloc[0]
        gaps = (diffs > most_common * 1.1).sum()
        return {"frequency": str(most_common), "gaps": int(gaps)}
    except Exception:
        return None


def build_profile_report(df, semantic_map):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    lines.append(f"Missing values: {int(df.isna().sum().sum())}")
    lines.append(f"Duplicates: {int(df.duplicated().sum())}")
    lines.append(f"Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}, Boolean: {len(bool_cols)}")

    if semantic_map:
        tagged = [f"{c} ({t})" for c, t in semantic_map.items()]
        lines.append("Semantic tags: " + ", ".join(tagged[:8]) + ("..." if len(tagged) > 8 else ""))

    skewed = []
    if numeric_cols:
        skew_vals = df[numeric_cols].skew().abs().sort_values(ascending=False)
        skewed = [f"{c} ({v:.2f})" for c, v in skew_vals.items() if v > 1][:5]
    if skewed:
        lines.append("Skewed numeric: " + ", ".join(skewed))

    high_card = [c for c in cat_cols if df[c].nunique() > df.shape[0] * 0.5]
    if high_card:
        lines.append("High-cardinality categoricals: " + ", ".join(high_card[:5]))

    corr_pairs = top_correlations(df, limit=3)
    if corr_pairs:
        pretty = [f"{a}↔{b}:{score:.2f}" for (a, b), score in corr_pairs]
        lines.append("Top correlations: " + ", ".join(pretty))

    return "\n".join(lines)


def render_ai_chart(df, chart_cfg):
    """Render a Plotly chart from the AI's suggestion when columns are valid."""
    chart_type = (chart_cfg.get("type") or "none").lower()
    x = chart_cfg.get("x")
    y = chart_cfg.get("y")
    hue = chart_cfg.get("hue")
    title = chart_cfg.get("title") or "AI suggested chart"

    valid = lambda col: col in df.columns if col else False
    fig = None

    if chart_type == "histogram" and valid(x):
        fig = px.histogram(df, x=x, color=hue if valid(hue) else None, title=title)
    elif chart_type == "bar" and valid(x):
        value_counts = df[x].value_counts().reset_index()
        value_counts.columns = [x, "Count"]
        fig = px.bar(value_counts, x=x, y="Count", color=hue if valid(hue) else None, title=title)
    elif chart_type == "box" and valid(x):
        fig = px.box(df, y=x, color=hue if valid(hue) else None, title=title)
    elif chart_type == "scatter" and valid(x) and valid(y):
        fig = px.scatter(df, x=x, y=y, color=hue if valid(hue) else None, title=title)

    return fig

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Agent", 
    layout="wide",
    page_icon="📊"
)

st.title("📊 Enhanced AI Data Analysis Agent")

# Google site verification for search console
st.markdown(
    '<meta name="google-site-verification" content="lpgOqVca5sBG-p8FFpw3YaXktCgH-q9zUnu0KrsIxto" />',
    unsafe_allow_html=True
)

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
        st.dataframe(df.head(10), width="stretch")
        
        # Data types information
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
        dtype_df.columns = ['Column', 'Data Type']
        dtype_df["Data Type"] = dtype_df["Data Type"].astype(str)
        st.dataframe(dtype_df, width="stretch")
        
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
            st.dataframe(missing_df, width="stretch")
            
            # Visualize missing values
            fig_missing = px.bar(missing_df, x='Column', y='Missing Count',
                               title='Missing Values by Column',
                               color='Missing Count')
            st.plotly_chart(fig_missing, width="stretch")
        else:
            st.success("🎉 No missing values found!")
    
    with tab2:
        st.header("Data Cleaning Tools")

        # Suggested cleaning actions based on the uploaded data
        cleaning_suggestions = generate_cleaning_suggestions(df)
        if cleaning_suggestions:
            st.subheader("Suggested Cleaning Actions")
            for s in cleaning_suggestions:
                st.markdown(f"- {s}")

            plan = build_cleaning_plan(df)
            preview_lines = []
            if plan.get("drop_duplicates"):
                preview_lines.append(f"Would drop {int(df.duplicated().sum())} duplicate rows")
            if plan.get("trim_whitespace"):
                preview_lines.append("Would trim whitespace in text columns")
            if plan.get("fill_missing"):
                preview_lines.append(f"Would fill missing values: total {int(df.isna().sum().sum())}")
            if plan.get("drop_constant"):
                drop_list = plan.get("drop_constant") or []
                preview_lines.append("Would drop constant numeric columns: " + ", ".join(drop_list))
            if preview_lines:
                st.caption("Planned actions:")
                for line in preview_lines:
                    st.markdown(f"- {line}")

            if st.button("Apply all suggested fixes"):
                df_clean = apply_cleaning_plan(df, plan)
                st.session_state.df = df_clean
                st.success("Applied suggested cleaning fixes.")
                st.rerun()
        else:
            st.success("Dataset looks clean—no immediate fixes suggested.")
        
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
                
                custom_value: str = ""
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
                            df_clean[col] = df_clean[col].ffill()
                        elif treatment_method == "Backward fill":
                            df_clean[col] = df_clean[col].bfill()
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
                    st.dataframe(duplicates, width="stretch")
            
            st.subheader("Column Operations")
            selected_column = st.selectbox("Select column for operation:", df.columns)
            
            if st.button("Show Unique Values"):
                unique_vals = df[selected_column].unique()
                st.write(f"Unique values in {selected_column}:")
                st.write(unique_vals)
            
            if st.button("Show Value Counts"):
                value_counts = df[selected_column].value_counts()
                st.write(f"Value counts for {selected_column}:")
                st.dataframe(value_counts, width="stretch")

            with st.expander("More data handling tools", expanded=False):
                # Trim whitespace for object columns
                object_cols = df.select_dtypes(include=["object"]).columns.tolist()
                if object_cols and st.button("Trim whitespace in text columns"):
                    df_clean = df.copy()
                    df_clean[object_cols] = df_clean[object_cols].apply(lambda col: col.str.strip())
                    st.session_state.df = df_clean
                    st.success("Whitespace trimmed in text columns.")
                    st.rerun()

                # Type conversion
                convert_col = st.selectbox("Convert column type", df.columns, key="convert_col")
                convert_target = st.selectbox("Target type", ["Numeric", "Datetime"], key="convert_target")
                if st.button("Apply conversion"):
                    df_clean = df.copy()
                    try:
                        if convert_target == "Numeric":
                            df_clean[convert_col] = pd.to_numeric(df_clean[convert_col], errors="coerce")
                        else:
                            df_clean[convert_col] = pd.to_datetime(df_clean[convert_col], errors="coerce")
                        st.session_state.df = df_clean
                        st.success("Column converted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Conversion failed: {exc}")

                # Outlier removal
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    outlier_col = st.selectbox("Remove outliers (IQR)", num_cols, key="outlier_col")
                    if st.button("Drop outliers"):
                        df_clean = df.copy()
                        q1 = df_clean[outlier_col].quantile(0.25)
                        q3 = df_clean[outlier_col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        before = len(df_clean)
                        df_clean = df_clean[(df_clean[outlier_col] >= lower) & (df_clean[outlier_col] <= upper)]
                        st.session_state.df = df_clean
                        st.success(f"Removed {before - len(df_clean)} outlier rows from {outlier_col}.")
                        st.rerun()

                    topk = st.number_input("Show top-k outlier rows", min_value=1, max_value=50, value=5, step=1, key="outlier_topk")
                    if st.button("View outlier rows"):
                        series = df[outlier_col]
                        q1 = series.quantile(0.25)
                        q3 = series.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        mask = (series < lower) | (series > upper)
                        out_df = df[mask].copy()
                        if out_df.empty:
                            st.info("No outliers detected for this column.")
                        else:
                            st.dataframe(out_df.head(int(topk)), width="stretch")

                # Normalize numeric column
                if num_cols:
                    norm_col = st.selectbox("Normalize column (min-max)", num_cols, key="norm_col")
                    if st.button("Normalize"):
                        df_clean = df.copy()
                        col_min = df_clean[norm_col].min()
                        col_max = df_clean[norm_col].max()
                        denom = col_max - col_min
                        if denom == 0:
                            st.warning("Cannot normalize: column has zero range.")
                        else:
                            df_clean[norm_col] = (df_clean[norm_col] - col_min) / denom
                            st.session_state.df = df_clean
                            st.success(f"Normalized {norm_col} to 0-1 range.")
                            st.rerun()

                # Standardize / log transform
                if num_cols:
                    trans_col = st.selectbox("Transform column", num_cols, key="trans_col")
                    trans_type = st.selectbox("Transform type", ["Standardize (z-score)", "Log1p"], key="trans_type")
                    if st.button("Apply transform"):
                        df_clean = df.copy()
                        if trans_type.startswith("Standardize"):
                            mean = df_clean[trans_col].mean()
                            std = df_clean[trans_col].std()
                            if std == 0:
                                st.warning("Cannot standardize: zero standard deviation.")
                            else:
                                df_clean[trans_col] = (df_clean[trans_col] - mean) / std
                                st.session_state.df = df_clean
                                st.success(f"Standardized {trans_col} (z-score).")
                                st.rerun()
                        else:
                            if (df_clean[trans_col] <= -1).any():
                                st.warning("Log1p requires values > -1; skipping.")
                            else:
                                df_clean[trans_col] = np.log1p(df_clean[trans_col])
                                st.session_state.df = df_clean
                                st.success(f"Applied log1p to {trans_col}.")
                                st.rerun()

                # Drop columns
                drop_cols = st.multiselect("Drop columns", df.columns, [])
                if drop_cols and st.button("Drop selected columns"):
                    df_clean = df.drop(columns=drop_cols)
                    st.session_state.df = df_clean
                    st.success(f"Dropped columns: {', '.join(drop_cols)}")
                    st.rerun()
    
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

                plot_options = ["Histogram", "Box Plot", "Density Plot", "Violin Plot"]
                if len(numeric_columns) > 1:
                    plot_options.append("Scatter vs")

                plot_type = st.selectbox("Select plot type:", plot_options)

                if plot_type == "Histogram":
                    fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, width="stretch")
                elif plot_type == "Box Plot":
                    fig = px.box(df, y=num_col, title=f"Box Plot of {num_col}")
                    st.plotly_chart(fig, width="stretch")
                elif plot_type == "Density Plot":
                    fig = px.histogram(df, x=num_col, marginal="rug", 
                                     title=f"Density Plot of {num_col}")
                    st.plotly_chart(fig, width="stretch")
                elif plot_type == "Violin Plot":
                    fig = px.violin(df, y=num_col, box=True, points="outliers", title=f"Violin Plot of {num_col}")
                    st.plotly_chart(fig, width="stretch")
                elif plot_type == "Scatter vs":
                    if len(numeric_columns) < 2:
                        st.info("Add another numeric column to plot a scatter chart.")
                    else:
                        other_numeric = [c for c in numeric_columns if c != num_col] or numeric_columns
                        num_col_y = st.selectbox("Select numeric column (Y):", other_numeric)
                        fig = px.scatter(df, x=num_col, y=num_col_y, title=f"Scatter of {num_col} vs {num_col_y}")
                        st.plotly_chart(fig, width="stretch")
            else:
                st.info("No numeric columns found for visualization.")
        
        with col2:
            st.subheader("Categorical Data Visualization")
            
            if categorical_columns:
                cat_col = st.selectbox("Select categorical column:", categorical_columns)
                
                plot_type = st.selectbox("Select categorical plot type:", 
                                       ["Bar Chart", "Pie Chart", "Treemap"])
                
                value_counts = df[cat_col].value_counts().reset_index()
                value_counts.columns = ['Category', 'Count']
                
                if plot_type == "Bar Chart":
                    fig = px.bar(value_counts, x='Category', y='Count', 
                               title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, width="stretch")
                elif plot_type == "Pie Chart":
                    fig = px.pie(value_counts, names='Category', values='Count',
                               title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, width="stretch")
                elif plot_type == "Treemap":
                    fig = px.treemap(value_counts, path=['Category'], values='Count', title=f"Treemap of {cat_col}")
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No categorical columns found for visualization.")
        
        # Correlation heatmap for numeric columns
        if len(numeric_columns) > 1:
            st.subheader("Correlation Heatmap")
            corr_matrix = df[numeric_columns].corr()
            fig = px.imshow(corr_matrix, title="Correlation Heatmap", 
                          aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, width="stretch")

        # Time-aware quick checks
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns.tolist()
        if datetime_cols:
            st.subheader("Time-aware checks")
            dt_col = st.selectbox("Datetime column", datetime_cols)
            gaps = time_gaps(df, dt_col)
            if gaps:
                st.markdown(f"- Inferred frequency: {gaps['frequency']}")
                st.markdown(f"- Gaps detected: {gaps['gaps']}")
            else:
                st.info("Could not infer frequency or no gaps detected.")
            if st.button("Plot rolling mean (7) on first numeric column"):
                num_cols_for_ts = df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols_for_ts:
                    df_sorted = df.sort_values(dt_col)
                    series = df_sorted[num_cols_for_ts[0]].rolling(window=7, min_periods=1).mean()
                    fig_ts = px.line(x=df_sorted[dt_col], y=series, title=f"7-step rolling mean of {num_cols_for_ts[0]}")
                    st.plotly_chart(fig_ts, width="stretch")
                else:
                    st.info("No numeric column to plot.")
    
    with tab4:
        st.header("AI-Powered Analysis")
        st.caption("Runs locally with heuristic suggestions—no API key needed.")

        use_llama = st.checkbox("Use local Llama (Ollama) for narrative", value=False)
        llama_model = st.text_input("Ollama model", value="llama3.1") if use_llama else ""

        if st.button("Run analysis", type="primary"):
            with st.spinner("Analyzing your data..."):
                analysis = generate_rule_based_insights(df)
                charts = generate_chart_suggestions(df)
                corr_pairs = top_correlations(df)
                outliers = outlier_counts(df)
                semantic_map = detect_semantic_types(df)
                profile_md = build_profile_report(df, semantic_map)

                llama_story = None
                if use_llama:
                    brief = build_data_brief(df)
                    prompt = (
                        "Craft a short, clear narrative about this dataset. "
                        "Mention shape, missingness, duplicates, and what to inspect first. "
                        "Keep it under 120 words.\n\n"
                        + json.dumps(brief)
                    )
                    llama_story = call_local_llama(prompt, model=llama_model or "llama3.1")

                if analysis.get("summary"):
                    st.subheader("Summary")
                    st.markdown(analysis["summary"])

                if profile_md:
                    st.subheader("Profile Snapshot")
                    st.markdown(profile_md)
                    st.download_button(
                        "Download snapshot", data=profile_md, file_name="profile_snapshot.md", mime="text/markdown"
                    )

                if llama_story:
                    st.subheader("Narrative (Llama)")
                    st.markdown(llama_story)

                data_fixes = analysis.get("data_fixes") or []
                if data_fixes:
                    st.subheader("Suggested Fixes")
                    for fix in data_fixes:
                        st.markdown(f"- {fix}")

                if corr_pairs:
                    st.subheader("Top correlations (abs)")
                    for (c1, c2), score in corr_pairs:
                        st.markdown(f"- {c1} ↔ {c2}: {score:.2f}")

                if outliers:
                    st.subheader("Columns with outliers (IQR)")
                    for col, count in outliers:
                        st.markdown(f"- {col}: {count} potential outliers")

                # Render primary chart from rule-based insights
                chart_cfg = analysis.get("chart") or {}
                fig = render_ai_chart(df, chart_cfg)
                if fig:
                    st.subheader("Suggested Chart")
                    if chart_cfg.get("reasoning"):
                        st.caption(chart_cfg.get("reasoning"))
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No suitable chart suggestion for the current data.")

                # Offer additional chart options
                if charts:
                    st.subheader("Other chart options")
                    for idx, cfg in enumerate(charts, start=1):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**{idx}. {cfg.get('title', 'Chart')}**")
                            if cfg.get("reasoning"):
                                st.caption(cfg["reasoning"])
                        with col_b:
                            if st.button(f"Show chart {idx}"):
                                fig_alt = render_ai_chart(df, cfg)
                                if fig_alt:
                                    st.plotly_chart(fig_alt, width="stretch")
                                else:
                                    st.info("This chart could not be rendered with current columns.")

        # Local Q&A (safe query)
        st.subheader("Ask the data (local)")
        st.caption("Use pandas query syntax, e.g., Age > 30 and Department == 'IT'")
        query_str = st.text_input("Filter query")
        agg_col = st.selectbox("Aggregate column (optional)", ["(none)"] + df.columns.tolist())
        agg_func = st.selectbox("Aggregation", ["count", "mean", "sum", "median", "max", "min"])
        if st.button("Run query"):
            try:
                filtered = df.query(query_str) if query_str else df
                st.write(f"Filtered rows: {len(filtered)}")
                if agg_col != "(none)" and pd.api.types.is_numeric_dtype(filtered[agg_col]):
                    val = getattr(filtered[agg_col], agg_func)()
                    st.write(f"{agg_func}({agg_col}) = {val}")
                st.dataframe(filtered.head(50), width="stretch")
            except Exception as exc:
                st.error(f"Query failed: {exc}")

        # Show current data state
        st.subheader("Current Data State")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head(), width="stretch")
        
        # Statistical Analysis
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width="stretch")
        
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
    csv = df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="📥 Download Cleaned Data as CSV",
        data=csv,
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Reset data button
    if st.sidebar.button("🔄 Reset to Original Data"):
        if uploaded_file is None:
            st.warning("No original uploaded file to reset from.")
        else:
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