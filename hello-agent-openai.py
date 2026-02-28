import os
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from config import API_KEY

# API key constant: will use environment variable OPENAI_API_KEY if available,
# otherwise falls back to the constant below.
# make sure there’s a key available
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY not set")
    st.stop()

def ask_agent(dfs: list[pd.DataFrame], prompt: str) -> str:
    llm = ChatOpenAI(openai_api_key=API_KEY, model="gpt-4o-mini", temperature=0.0)
    agent = create_pandas_dataframe_agent(
        llm,
        dfs,
        verbose=True,
        allow_dangerous_code=True,
    )
    return agent.invoke(prompt)["output"]


st.title("Hello Agent (OpenAI)")

file_paths_input = st.text_input(
    "Enter file path(s) — separate multiple paths with a comma",
    key="file_paths",
)
prompt = st.text_input("Enter a prompt", key="prompt")

# Inputs are only processed when this button is pressed
if st.button("Run"):
    dfs = []
    if file_paths_input:
        paths = [p.strip() for p in file_paths_input.split(",") if p.strip()]
        for path in paths:
            try:
                df = pd.read_csv(path)
                st.subheader(path)
                st.write(df)
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
        st.info("Press Run after entering file path(s) and a prompt.")