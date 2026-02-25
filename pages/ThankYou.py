import streamlit as st


def show() -> None:
	"""Render the Thank You page. Call `show()` from another module to display."""
	st.title("Thank You")
	st.write("Thanks for exploring CycleOps.")
	st.write("We appreciate your interest and feedback.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
