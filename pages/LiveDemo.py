import streamlit as st


def show() -> None:
	"""Render the Live demo page. Call `show()` from another module to display."""
	st.title("Live Demo")
	st.write("Interactive demonstration of the live data and visualizations.")
	st.write("Embed charts, maps, or live feeds here for users to explore.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
