import streamlit as st
import requests
from streamlit_oauth import OAuth2Component
from my_search import query_faiss  # your semantic search module

# ----- Configuration -----
CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID"
CLIENT_SECRET = "YOUR_GOOGLE_CLIENT_SECRET"
REDIRECT_URI = "http://localhost:8501"
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USER_INFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# ----- OAuth Setup -----
oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_ENDPOINT,
    token_endpoint=TOKEN_ENDPOINT,
)

# ----- Streamlit Page Setup -----
st.set_page_config(page_title="📚 BabyLLM", page_icon="📚")
st.title("📚 BabyLLM")

# ----- Preserve OAuth Token in Session State -----
if "token" not in st.session_state:
    token = oauth2.authorize_button(
        name="Continue with Google",
        redirect_uri=REDIRECT_URI,
        scope="profile email"
    )
    if token:
        st.session_state.token = token
else:
    token = st.session_state.token

# ----- Post-Login Section -----
if token:
    if "token" in token and "access_token" in token["token"]:
        access_token = token["token"]["access_token"]
    else:
        st.error("Access token not found.")
        st.stop()

    response = requests.get(USER_INFO_URL, headers={"Authorization": f"Bearer {access_token}"})
    user_info = response.json()
    email = user_info["email"]
    name = user_info["name"]

    st.success(f"Welcome, {name}!")

    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.info(f"Queries used: {st.session_state.query_count}/3")

    # If query count is less than 3, allow prompt submission.
    if st.session_state.query_count < 3:
        query = st.text_input("Ask a question:")
        if st.button("Submit"):
            answer = query_faiss(query)
            st.session_state.chat_history.append((query, answer))
            st.session_state.query_count += 1
            st.success("Query submitted!")
    else:
        # Show a modal pop-up that stops further background processing.
        with st.modal("Subscription Required"):
            st.write("You have reached your free prompt limit. Please subscribe to continue using BabyLLM.")
            if st.button("Subscribe Now"):
                st.write("Redirecting to subscription page...")
                # Here you would add your subscription/payment integration.
        st.stop()  # Stop further rendering.

    st.subheader("Previous Queries and Answers")
    for idx, (q, a) in enumerate(st.session_state.chat_history, start=1):
        st.markdown(f"**Query {idx}:** {q}")
        st.markdown(f"**Answer {idx}:** {a}")
        st.markdown("---")
else:
    st.info("Please log in with Google to continue.")
