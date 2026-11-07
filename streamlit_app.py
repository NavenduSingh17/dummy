import streamlit as st
from uuid import uuid4

from auth import authenticate_user
from chat_agent import chat


st.set_page_config(
    page_title="FinSolve AI Assistant",
    page_icon="💬",
    layout="centered",
)


def login():
    st.title("FinSolve AI Assistant")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        user = authenticate_user(
            username.strip(),
            password,
        )

        if user is None:
            st.error("Invalid username or password.")
            return

        # The role comes from SQLite, not from the chat input.
        st.session_state["authenticated"] = True
        st.session_state["username"] = user["username"]
        st.session_state["user_role"] = user["user_role"]

        # A fresh conversation for every successful login.
        st.session_state["thread_id"] = str(uuid4())
        st.session_state["messages"] = []

        st.rerun()


def logout():
    # Discard this UI session's identity and conversation.
    for key in [
        "authenticated",
        "username",
        "user_role",
        "thread_id",
        "messages",
    ]:
        st.session_state.pop(key, None)

    st.rerun()


def chat_page():
    username = st.session_state["username"]
    user_role = st.session_state["user_role"]

    with st.sidebar:
        st.title("Account")
        st.write(f"**Username:** {username}")
        st.write(f"**Role:** {user_role}")

        st.button(
            "Logout",
            on_click=logout,
            use_container_width=True,
        )

    st.title("FinSolve AI Assistant")
    st.caption("Ask questions about company documents and employee data.")

    # Render the conversation maintained by this Streamlit session.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question...")

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = chat(
                        user_question=question,
                        user_role=user_role,
                        session_thread_id=st.session_state["thread_id"],
                    )
                except Exception:
                    # Log the full exception server-side when debugging.
                    # st.exception(Exception("The assistant encountered an error."))
                    st.error("The assistant encountered an error. Check the terminal for details.")
                    st.stop()

                st.markdown(answer)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
        })


if not st.session_state.get("authenticated", False):
    login()
else:
    chat_page()
