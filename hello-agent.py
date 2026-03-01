import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from config import API_KEY_GEMINI

if not API_KEY_GEMINI:
    API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")
if not API_KEY_GEMINI:
    st.error("API_KEY_GEMINI not set")
    st.stop()

def ask_agent(dfs: list[pd.DataFrame], prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(google_api_key=API_KEY_GEMINI, model="gemini-2.5-flash")
    agent = create_pandas_dataframe_agent(
        llm,
        dfs,  # pass a list of DataFrames
        verbose=True,
        allow_dangerous_code=True,
    )
    return agent.invoke(prompt)["output"]


st.title("Hello Agent")

file_paths_input = st.text_input(
    "Enter file path(s) — separate multiple paths with a comma",
    key="file_paths",
)
prompt = st.text_input("Enter a prompt", key="prompt")

if st.button("Run"):
    dfs = []
    if file_paths_input:
        paths = [p.strip() for p in file_paths_input.split(",") if p.strip()]
        for path in paths:
            try:
                df = pd.read_csv(path)
                st.subheader(path)
                st.write(df.head())  # Show only the first few rows for brevity
                dfs.append(df)
            except Exception as e:
                st.error(f"Error reading '{path}': {e}")

    if prompt and dfs:
        with st.spinner("Thinking..."):
            try:
                response = ask_agent(dfs, prompt)
                st.write(response)
            except Exception as e:
                st.error(f"Error from agent: {e}")
    elif prompt and not dfs:
        st.warning("Please enter at least one valid file path first.")
    else:
        st.warning("Please enter a prompt and at least one file path.")