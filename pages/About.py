import streamlit as st


def show() -> None:
	"""Render the About page. Call `show()` from another module to display."""
	st.title("About")
	st.write("Made as part of our group project for CET 522")
	st.write("Created by Danyel Redd, Dave Carpenter and Sophie De Rosa")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
	
