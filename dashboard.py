import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.title("Youtube Comment Analysis")

link = st.text_input("Enter Youtube Video Link: ")

if st.button("Submit"):
    if link:
        with st.spinner("Fetching comments and analyzing..."):
            response = requests.post("http://127.0.0.1:8000/analyze", params = {"youtube_link": link})

            if response.status_code == 200:
                data = response.json()
                sentiment = data.get("Sentiment", {})
                emotion = data.get("Emotion", {})
                adjectives = data.get("Adjectives", [])
                
            
            sentiment_data = pd.DataFrame(list(sentiment.items()), columns=['Sentiment', 'Percentage'])
            emotion_data = pd.DataFrame(list(emotion.items()), columns=['Emotion', 'Percentage'])

            # pie chart
            fig, ax = plt.subplots()
            ax.pie(sentiment_data["Percentage"], labels=sentiment_data["Sentiment"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            # bar chart for emotion distribution
            st.subheader("Emotion Distribution")

            fig, ax = plt.subplots()
            ax.bar(emotion_data["Emotion"], emotion_data["Percentage"])
            ax.set_xlabel("Emotions")
            ax.set_ylabel("Percentage (%)")
            ax.set_title("Emotion Analysis of Comments")

            # optional: make it more readable
            for i, v in enumerate(emotion_data["Percentage"]):
                ax.text(i, v + 1, f"{v}%", ha='center')

            st.pyplot(fig)

            # word cloud
            wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(adjectives))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

    else:
        st.error("Error: Could not fetch sentiment data")
