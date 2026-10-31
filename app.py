"""XRD Mineral Matcher: identify likely minerals from a powder XRD pattern with k-NN on the RRUFF library.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="XRD Mineral Matcher", page_icon=":material/diamond:", layout="wide")

import ui  # noqa: E402  (set_page_config must run first)

ui.setup_page()

pages = [
    st.Page("views/identify.py", title="Identify", icon=":material/search:", default=True),
    st.Page("views/performance.py", title="Performance", icon=":material/monitoring:"),
    st.Page("views/library.py", title="Library", icon=":material/library_books:"),
    st.Page("views/how_it_works.py", title="How It Works", icon=":material/info:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.caption("A STEM project using k-nearest neighbors on measured mineral patterns from the RRUFF Project.")

nav.run()
