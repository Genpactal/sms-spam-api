import requests
import streamlit as st


st.title("SMS Spam Detector")
st.write("Enter an English SMS message to check whether it looks like spam.")
message = st.text_area("Message")

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"message": message}, timeout=10,
            )
            response.raise_for_status()
            if response.json()["prediction"] == "spam":
                st.error("Spam")
            else:
                st.success("Not spam")
        except requests.RequestException:
            st.error("Cannot reach the prediction API. Start FastAPI on port 8000.")
