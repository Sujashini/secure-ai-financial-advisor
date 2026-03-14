import streamlit as st
from backend.users.service import change_password


def render_profile_page(user):
    st.subheader("👤 Profile & Settings")
    st.markdown(f"**Username:** {user.username}")
    st.markdown(f"**Email:** {user.email}")

    st.markdown("---")
    st.markdown("### Change password")

    old_pw = st.text_input("Current password", type="password", key="prof_old_pw")
    new_pw1 = st.text_input("New password", type="password", key="prof_new_pw1")
    new_pw2 = st.text_input("Confirm new password", type="password", key="prof_new_pw2")

    if st.button("Update password", key="btn_change_pw"):
        if not old_pw or not new_pw1 or not new_pw2:
            st.warning("Please fill in all the fields.")
        elif new_pw1 != new_pw2:
            st.error("New passwords do not match.")
        elif len(new_pw1) < 8:
            st.error("New password must be at least 8 characters long.")
        else:
            try:
                change_password(user.id, old_pw, new_pw1)
                st.success("Password updated successfully.")
            except ValueError as e:
                st.error(str(e))
            except Exception:
                st.error("Something went wrong while updating your password. Please try again.")