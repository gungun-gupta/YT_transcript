import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Summarization pipeline using a lighter model
summarization_pipeline = pipeline("summarization", model="t5-small")

# Helper functions
def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r"v=([^&]+)",       # URL with 'v=' parameter
        r"youtu.be/([^?]+)", # Shortened URL
        r"youtube.com/embed/([^?]+)" # Embedded URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def summarize_text(text, max_length=200):
    """Summarize text using a smaller pre-trained model."""
    if len(text) > 512:
        text = text[:512]  # Truncate text to avoid exceeding model limits
    summary = summarization_pipeline(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def extract_keywords(text):
    """Extract top keywords from text."""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    return list(set(keywords))[:5]

# Main Streamlit app
def main():
    st.title("YouTube Video Summarizer")
    st.markdown("""
        This app extracts and summarizes the transcript of a YouTube video. 
        If the video has no transcript, you can input the text manually.
    """)

    # User input for YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:", "")

    # User customization options
    max_summary_length = st.slider("Max Summary Length (in characters):", 100, 1000, 200)

    if st.button("Summarize"):
        try:
            # Extract video ID from URL
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            try:
                # Fetch transcript using YouTubeTranscriptApi
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                video_text = " ".join([line["text"] for line in transcript])
            except (TranscriptsDisabled, NoTranscriptFound):
                st.warning("Transcripts are unavailable for this video.")
                manual_input = st.text_area("Please paste the transcript manually:")
                if manual_input.strip():
                    video_text = manual_input
                else:
                    st.error("No transcript provided. Please paste the transcript to proceed.")
                    return

            # Summarize the transcript
            summary = summarize_text(video_text, max_length=max_summary_length)

            # Extract keywords
            keywords = extract_keywords(video_text)

            # Perform sentiment analysis
            sentiment = TextBlob(video_text).sentiment

            # Display results
            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(", ".join(keywords))

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
