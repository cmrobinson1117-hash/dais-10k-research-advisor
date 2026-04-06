import streamlit as st

from agent import ask_agent


st.set_page_config(page_title="SEC 10-K Research Advisor", layout="wide")
st.title("SEC 10-K Research Advisor")
st.caption("Agentic prototype for asking questions over SEC 10-K filings")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask about risk factors, company comparisons, or trends across the filings. "
                "Example: Compare Apple and Tesla cybersecurity risk disclosures in 2024."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question about the SEC 10-K corpus")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching filings and generating answer..."):
            try:
                response = ask_agent(prompt)
            except Exception as exc:
                response = f"Error: {exc}"
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
