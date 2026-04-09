import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Stopwords
stop_words = set(stopwords.words('english'))

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💬 Opinion Mining & Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("### Analyze customer reviews instantly ")

# Input box
text = st.text_area("✍️ Enter your review here:")

# Predict button
if st.button("🔍 Analyze Sentiment"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        text_clean = clean_text(text)
        text_vec = vectorizer.transform([text_clean])

        result = model.predict(text_vec)[0]
        prob = model.predict_proba(text_vec).max()

        # Output styling
        if result == "positive":
            st.success(f"😊 Positive Sentiment (Confidence: {prob:.2f})")
        elif result == "negative":
            st.error(f"😡 Negative Sentiment (Confidence: {prob:.2f})")
        else:
            st.info(f"😐 Neutral Sentiment (Confidence: {prob:.2f})")

# Divider
st.markdown("---")

# Extra Features
st.subheader("Quick Tips")
st.write("""
- Positive 😊 → Good reviews  
- Neutral 😐 → Mixed feedback  
- Negative 😡 → Bad experience  
""")

# Footer
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)