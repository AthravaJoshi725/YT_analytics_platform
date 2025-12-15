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

if "video_details" not in st.session_state:
    st.session_state.video_details = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None


# --------------------------
# INPUT: YOUTUBE LINK
# --------------------------
yt_link = st.text_input("Enter YouTube Video Link:")

if st.button("Analyze"):
    if not yt_link:
        st.error("Please enter a YouTube link.")
    else:
        with st.spinner("Extracting comments + creating RAG..."):
            res = requests.post(f"{API_URL}/analyze", params={"youtube_link": yt_link})

            if res.status_code == 200:
                response = res.json()

                st.session_state.video_details = response
                st.session_state.video_id = response.get("video_id")   
                st.session_state.rag_ready = True
                st.session_state.chat_history = []

                st.success("RAG is ready! You can now ask questions.")
            else:
                st.error("Failed to analyze the video.")


# --------------------------
# SHOW VIDEO DETAILS
# --------------------------
if st.session_state.video_details:
    vd = st.session_state.video_details

    st.markdown("### 🎬 Video Information")
    st.image(vd.get("thumbnail"), width=400)

    st.write(f"**Title:** {vd.get('title')}")
    st.write(f"**Channel:** {vd.get('channelName')}")
    st.write(f"**Published:** {vd.get('publishedAt')}")
    # st.write(f"**Description:** {vd.get('description')}")


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
                        "video_id": st.session_state.video_id,   
                        "user_query": user_query
                    }
                )

                if res.status_code == 200:
                    answer = res.text.strip('"')
                    st.session_state.chat_history.append(("You", user_query))
                    st.session_state.chat_history.append(("AI", answer))
                else:
                    st.error("Backend error or RAG not ready.")


# --------------------------
# DISPLAY CHAT HISTORY
# --------------------------
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")
