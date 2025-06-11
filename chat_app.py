import streamlit as st
import requests

# initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Chat with MeGPT")

# allow overriding API URL via sidebar
api_url = st.sidebar.text_input("API URL", "http://localhost:8000/generate/")

# handle user input first
user_input = st.chat_input("Your message")
if user_input:
    # immediately record & render the user message
    st.session_state.history.append((user_input, "user"))
    st.chat_message("user").write(user_input)

    # call FastAPI with spinner animation
    with st.spinner(""):
        try:
            resp = requests.post(api_url, json={"prompt": user_input})
            resp.raise_for_status()
            bot_reply = resp.json().get("generated_text", "")
        except Exception as e:
            bot_reply = f"Error: {e}"

    # render assistant response and save to history
    st.chat_message("assistant").write(bot_reply)
    st.session_state.history.append((bot_reply, "assistant"))

    # stop so we don't re‐render the history twice
    st.stop()

# now display the full chat history on normal runs
for text, sender in st.session_state.history:
    if sender == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)