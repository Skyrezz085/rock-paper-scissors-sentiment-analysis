# Libraries
import streamlit as st
import eda
import prediction

# Navigation section
navigation = st.sidebar.selectbox("Choose Page", ("Sentiment Analysis","EDA"))

# Page
if navigation == "Sentiment Analysis":
    prediction.run()
else:
    eda.run()