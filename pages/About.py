import streamlit as st


def show() -> None:
	"""Render the About page. Call `show()` from another module to display."""
	st.title("About")
	st.write("Made as part of our group project for CET 522")
	st.write("Created by Dave Carpenter, Sophie De Rosa, and Danyel Redd")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
	
