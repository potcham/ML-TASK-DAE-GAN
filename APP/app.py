import streamlit as st
from src.page1 import page1
from src.page2 import page2

pages = {
    "Entry point": page1,
    "Text to image": page2,
}

# Create the selectbox in the sidebar
page = st.sidebar.selectbox("Select a page", list(pages.keys()))

# Display the selected page
pages[page]()