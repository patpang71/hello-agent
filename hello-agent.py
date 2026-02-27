import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent


def ask_agent(dfs: list[pd.DataFrame], prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
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
