import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():

    st.title("Email Spam Detection")

    st.write("This app predicts whether an email is Spam or Not Spam.")

    # Load CSV
    df = pd.read_csv("email_spam_detection_cleaned.csv")

    # Features and Target
    X = df[[
        "Email_Length",
        "Num_Links",
        "Num_Special_Chars",
        "Capital_Words",
        "Has_Attachment"
    ]]
    y = df["Spam"]

    # Train Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # User Inputs
    email_length = st.number_input("Email Length", min_value=0, value=100)
    num_links = st.number_input("Number of Links", min_value=0, value=1)
    num_special = st.number_input("Number of Special Characters", min_value=0, value=5)
    capital_words = st.number_input("Capital Words", min_value=0, value=2)
    attachment = st.selectbox("Has Attachment", ["No", "Yes"])

    has_attachment = 1 if attachment == "Yes" else 0

    input_data = pd.DataFrame({
        "Email_Length": [email_length],
        "Num_Links": [num_links],
        "Num_Special_Chars": [num_special],
        "Capital_Words": [capital_words],
        "Has_Attachment": [has_attachment]
    })

    if st.button("Predict"):

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam Email")

if __name__ == "__main__":
    main()