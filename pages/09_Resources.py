import streamlit as st


def show() -> None:
	"""Render the Resources page. Call `show()` from another module to display."""
	st.title("Resources")
	st.write("Citibike System Data: Lyft, Inc. (2026). Citi Bike system data. https://citibikenyc.com/system-data")
	st.write("Lyft, Inc. (n.d.). How it works. Citi Bike. Retrieved March 7, 2026, from https://citibikenyc.com/how-it-works")
	st.write("We used GitHub Copilot to assist with coding tasks and building the app.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
