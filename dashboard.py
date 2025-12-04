import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("YouTube RAG Assistant")

# --------------------------
# SESSION STATE
# --------------------------
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------
# INPUT: YOUTUBE LINK
# --------------------------
yt_link = st.text_input("Enter YouTube Video Link:")

if st.button("Analyze"):
    if not yt_link:
        st.error("Please enter a YouTube link.")
    else:
        with st.spinner("Extracting comments and building RAG..."):
            res = requests.post(f"{API_URL}/analyze", params={"youtube_link": yt_link})

            if res.status_code == 200 and res.json() == True:
                st.success("RAG is ready! You can now ask questions.")
                st.session_state.rag_ready = True
                st.session_state.chat_history = []  # reset chat
            else:
                st.error("Failed to create RAG. Try again later.")


# --------------------------
# CHAT BOX (Only show if RAG is ready)
# --------------------------
if st.session_state.rag_ready:

    st.subheader("Ask Questions Based on YouTube Comments")

    user_query = st.text_input("Your question:")

    if st.button("Send"):
        if not user_query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                res = requests.post(
                    f"{API_URL}/ask",
                    params={
                        "youtube_link": yt_link,
                        "user_query": user_query
                    }
                )

                if res.status_code == 200:
                    answer = res.text.strip('"')
                    st.session_state.chat_history.append(("You", user_query))
                    st.session_state.chat_history.append(("AI", answer))
                else:
                    st.error("RAG not ready or backend error.")


# --------------------------
# DISPLAY CHAT HISTORY
# --------------------------
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")
