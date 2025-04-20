# import streamlit as st
# from auth import login_user, register_user
# from utils import get_logged_user

# st.set_page_config("Login Demo", layout="centered")

# menu = st.sidebar.selectbox("Menu", ["Login", "Register", "Dashboard"])

# if menu == "Register":
#     st.title("Register")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Register"):
#         if register_user(username, password):
#             st.success("Registered! Go to Login")
#         else:
#             st.error("User already exists!")

# elif menu == "Login":
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         token = login_user(username, password)
#         if token:
#             st.session_state["token"] = token
#             st.success("Logged in!")
#         else:
#             st.error("Invalid credentials")

# elif menu == "Dashboard":
#     user = get_logged_user()
#     if not user:
#         st.warning("Please login first")
#     else:
#         st.success(f"Welcome, {user['username']} ({user['role']})")

#         if user["role"] == "admin":
#             st.write("🛡️ This is the Admin Panel")
#         elif user["role"] == "user":
#             st.write("👤 This is the User Dashboard")

#         if st.button("Logout"):
#             st.session_state.pop("token")
#             st.success("Logged out")




import streamlit as st
from auth import login_user, register_user
from utils import get_logged_user

# Set page config
st.set_page_config(page_title="Login Demo", layout="centered")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "token" not in st.session_state:
    st.session_state.token = None

def show_login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            token = login_user(username, password)
            if token:
                st.session_state.token = token
                st.session_state.page = "Dashboard"
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with col2:
        if st.button("Go to Register"):
            st.session_state.page = "Register"
            st.rerun()

def show_register_page():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registered successfully! Please go to Login.")
        else:
            st.error("User already exists!")
    if st.button("Back to Login"):
        st.session_state.page = "Login"
        st.rerun()

def show_dashboard():
    user = get_logged_user()
    if not user or not st.session_state.token:
        st.warning("Please login first")
        st.session_state.page = "Login"
        st.session_state.token = None
        st.rerun()
        return

    # Import and run the surveillance dashboard
    import streamlit1  # Importing streamlit1.py
    streamlit1.start_app(user)  # Call start_app with user parameter

def main():
    if st.session_state.page == "Login":
        show_login_page()
    elif st.session_state.page == "Register":
        show_register_page()
    elif st.session_state.page == "Dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()