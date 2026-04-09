import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stopwords (no NLTK needed)
stop_words = ENGLISH_STOP_WORDS

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # keep only letters
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>💬 Opinion Mining & Sentiment Analysis</h1>",
    unsafe_allow_html=True
)

st.write("### Analyze customer reviews instantly 🚀")

# Input box
text = st.text_area("✍️ Enter your review here:")

# Button
if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess
        text_clean = clean_text(text)
        text_vec = vectorizer.transform([text_clean])

        # Prediction
        result = model.predict(text_vec)[0]

        # Confidence (if available)
        try:
            prob = model.predict_proba(text_vec).max()
        except:
            prob = None

        # Output
        if result in ["positive", 1]:
            if prob:
                st.success(f"😊 Positive Sentiment (Confidence: {prob:.2f})")
            else:
                st.success("😊 Positive Sentiment")

        elif result in ["negative", 0]:
            if prob:
                st.error(f"😡 Negative Sentiment (Confidence: {prob:.2f})")
            else:
                st.error("😡 Negative Sentiment")

        else:
            if prob:
                st.info(f"😐 Neutral Sentiment (Confidence: {prob:.2f})")
            else:
                st.info("😐 Neutral Sentiment")

# Divider
st.markdown("---")

# Tips Section
st.subheader("💡 Quick Tips")
st.write("""
- Positive 😊 → Good reviews  
- Neutral 😐 → Mixed feedback  
- Negative 😡 → Bad experience  
""")

# Footer
st.markdown(
    "<p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)