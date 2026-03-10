import streamlit as st


def show() -> None:
	"""Render the Recommendations page. Call `show()` from another module to display."""
	st.title("Recommendations")
	st.write("Investigate inter-neighborhood infrastructure demands")
	st.write("Explore winter pricing and winterized infrastructure to incentivize year-round ridership")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
