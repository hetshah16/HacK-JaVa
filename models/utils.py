import streamlit as st
from auth import decode_token

def get_logged_user():
    token = st.session_state.get("token")
    if not token:
        return None
    return decode_token(token)
