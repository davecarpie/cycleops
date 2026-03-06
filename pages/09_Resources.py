import streamlit as st


def show() -> None:
	"""Render the Resources page. Call `show()` from another module to display."""
	st.title("Resources")
	st.write("streamlit.io")
	st.write("Provide citations, data access instructions, and further reading.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
