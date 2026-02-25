import streamlit as st


def show() -> None:
	"""Render the Background page. Call `show()` from another module to display."""
	st.title("Background")
	st.write("Context and motivation for the CycleOps project.")
	st.write("Describe data sources, goals, and high-level approach here.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
