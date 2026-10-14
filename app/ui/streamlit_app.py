"""Local application entrypoint. All research/model APIs remain in their existing modules."""

from pathlib import Path

import streamlit as st

from app.ui.design import CSS
from app.ui.helpers import PAGES, default_workspace
from app.ui.pages import dataset_page, fairness_page, inspect_page, model_page

st.set_page_config(
    page_title="SkinNet+ · Dermatology research", layout="wide", initial_sidebar_state="collapsed"
)
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="topbar"><div class="brand">SkinNet+<small>Dermatology research</small></div><span class="prototype">Academic research prototype</span></div>',
    unsafe_allow_html=True,
)
root = Path(st.session_state.get("_artifact_root", "artifacts"))
if "workspace" not in st.session_state:
    st.session_state["workspace"] = default_workspace(root)
page = st.radio(
    "Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="navigation"
)
workspace = st.session_state["workspace"]
if page == "Inspect an image":
    inspect_page(workspace)
elif page == "Dataset & evaluation":
    dataset_page(workspace)
elif page == "Fairness research":
    fairness_page(workspace)
else:
    model_page(workspace, root)
st.markdown(
    '<div class="footer">Academic research prototype · not clinically validated and not a substitute for professional medical evaluation.</div>',
    unsafe_allow_html=True,
)
