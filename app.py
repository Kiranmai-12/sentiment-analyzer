import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
st.title("Movie Review Sentiment Analyzer")
x=["loved this wonderful movie","excellent exciting story","hated this boring movie","terrible and disappointing","really enjoyable","waste of time"]
y=["Positive","Positive","Negative","Negative","Positive","Negative"]
model=make_pipeline(TfidfVectorizer(),LogisticRegression()).fit(x,y)
review=st.text_area("Enter a review")
if st.button("Analyze") and review.strip(): st.success(model.predict([review])[0])
st.caption("Tiny sample model for learning; results may be inaccurate.")
