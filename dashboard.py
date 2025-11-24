import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

API_URL = "http://127.0.0.1:8000"

st.title("YouTube Comment Analyzer + RAG Q&A")

st.write("Enter a YouTube link to analyze comments, then ask AI questions based only on the comments.")

link = st.text_input("Enter YouTube Video Link:")

# Store analysis results and DB availability
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if st.button("Analyze Video"):
    if not link:
        st.error("Please paste a valid YouTube video link.")
    else:
        with st.spinner("Extracting comments and performing analysis..."):
            response = requests.post(f"{API_URL}/analyze", params={"youtube_link": link})

            if response.status_code == 200:
                st.session_state.analysis_done = True
                data = response.json()

                sentiment = data.get("Sentiment", {})
                emotion = data.get("Emotion", {})
                adjectives = data.get("Adjectives", [])

                st.success("Analysis completed!")

                # Sentiment Pie Chart
                sentiment_data = pd.DataFrame(list(sentiment.items()), columns=["Sentiment", "Percentage"])
                fig, ax = plt.subplots()
                ax.pie(sentiment_data["Percentage"], labels=sentiment_data["Sentiment"],
                       autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.subheader("Sentiment Distribution")
                st.pyplot(fig)

                # Emotion Bar Chart
                emotion_data = pd.DataFrame(list(emotion.items()), columns=["Emotion", "Percentage"])
                st.subheader("Emotion Distribution")
                fig, ax = plt.subplots()
                ax.bar(emotion_data["Emotion"], emotion_data["Percentage"])
                ax.set_xlabel("Emotions")
                ax.set_ylabel("Percentage (%)")
                for i, v in enumerate(emotion_data["Percentage"]):
                    ax.text(i, v + 1, f"{v}%", ha='center')
                st.pyplot(fig)

                # Word Cloud
                st.subheader("Word Cloud (Adjectives)")
                wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(adjectives))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

            else:
                st.error("Error: Could not fetch sentiment data.")


# ---------------------------
# RAG Question Answering
# ---------------------------
st.subheader("Ask AI a Question (Based on Comments Only)")

if st.session_state.analysis_done:
    user_question = st.text_input("Ask something based on the comments:")

    if st.button("Ask"):
        if not user_question:
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                rag_response = requests.post(
                    f"{API_URL}/ask",
                    params={
                        "youtube_link": link,
                        "user_query": user_question
                    }
                )

                if rag_response.status_code == 200:
                    answer = rag_response.text.strip('"')
                    st.success("AI Answer:")
                    st.write(answer)
                else:
                    st.error("Failed to generate answer. Check backend logs.")
else:
    st.info("Run analysis first before asking questions.")
