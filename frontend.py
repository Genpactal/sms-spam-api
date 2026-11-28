import os

import requests
import streamlit as st


st.title("SMS Spam Detector")
st.write("Enter an English SMS message to check whether it looks like spam.")
api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
message = st.text_area("Message")

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={"message": message}, timeout=10,
            )
            response.raise_for_status()
            if response.json()["prediction"] == "spam":
                st.error("Spam")
            else:
                st.success("Not spam")
        except requests.RequestException:
            st.error("Cannot reach the prediction API. Check that the API is running.")
