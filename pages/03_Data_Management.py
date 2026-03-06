import streamlit as st
from pathlib import Path


def show() -> None:
    """Render the Data Management page. Call `show()` from another module to display."""
    st.title("Data Management")

    # Show ER diagram image if available. Try the page folder first, then repo root.
    img_path = Path(__file__).parent / "ER Diagram Citibikes.png"
    if not img_path.exists():
        img_path = Path(__file__).parent.parent / "ER Diagram Citibikes.png"

    if img_path.exists():
        # `use_column_width` is deprecated; use `width` instead.
        # Provide a reasonable default pixel width for layout.
        st.image(str(img_path), caption="ER Diagram — Citibikes", width=700)
    else:
        st.warning("ER Diagram Citibikes.png not found in the repo.")


# When Streamlit runs this file as its own page, render the UI immediately.
show()
