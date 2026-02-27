# hello-agent

A simple Streamlit app that accepts a CSV file path and a text prompt as inputs.

## Prerequisites

Make sure you have Python installed. Then install the required dependencies:

```bash
pip install streamlit pandas
```

## Running the App

```bash
streamlit run hello-agent.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Usage

1. **File Path** — Enter the full path to a CSV file (e.g. `/Users/you/data.csv`)
2. **Prompt** — Enter any text prompt

The app will display the CSV as a table and echo back your prompt.