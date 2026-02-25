import streamlit as st


def show() -> None:
	"""Render the Predictive model page. Call `show()` from another module to display."""
	st.title("Predictive Model")
	st.write("Overview of the predictive modeling approach and results.")
	st.write("Include model descriptions, evaluation metrics, and examples.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
