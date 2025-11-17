# AI Data Analysis Agent - Development Guide

This guide provides essential context for AI agents working with this codebase.

## Project Overview

This is a Streamlit-based web application that enables AI-powered data analysis through natural language queries. The application uses PandasAI with OpenAI's language model to interpret questions and analyze data from uploaded CSV/Excel files.

## Key Components

- **UI Layer** (`data_agent.py`): Streamlit-based interface handling file uploads and user interactions
- **Analysis Engine**: PandasAI integration with OpenAI for natural language data analysis
- **Data Processing**: Pandas for data manipulation and matplotlib for visualization

## Critical Workflows

### Development Setup

1. Dependencies required:
   - streamlit
   - pandas
   - pandasai
   - matplotlib
   - openai

2. Environment configuration:
   - OpenAI API key must be configured (currently hardcoded, should be moved to environment variables)

### Key Patterns

1. **Data Loading Pattern**:
   ```python
   if uploaded_file.name.endswith(".csv"):
       df = pd.read_csv(uploaded_file)
   else:
       df = pd.read_excel(uploaded_file)
   ```

2. **Analysis Pattern**:
   - Convert Pandas DataFrame to SmartDataframe
   - Use chat() method for natural language analysis
   - Handle both text and matplotlib Figure outputs

3. **Error Handling**:
   ```python
   try:
       result = smart_df.chat(question)
   except Exception as e:
       st.error(f"Error: {e}")
   ```

## Integration Points

1. **OpenAI Integration**:
   - Uses PandasAI's OpenAI wrapper
   - Requires valid API token
   - Handles async communication with OpenAI's API

2. **File Format Support**:
   - CSV (.csv)
   - Excel (.xlsx)

## Areas for Improvement

1. Security:
   - Move API key to environment variables
   - Add input validation for file uploads

2. Performance:
   - Consider caching for repeated analyses
   - Add progress indicators for long-running analyses

## Common Tasks

1. Adding new file format support:
   - Extend file_uploader type list
   - Add corresponding pandas read_* function

2. Implementing new visualizations:
   - Ensure compatibility with matplotlib figures
   - Handle new output types in the result handler