"""Local application entrypoint. All research/model APIs remain in their existing modules."""

from pathlib import Path

import streamlit as st
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.ui.design import CSS
from app.ui.helpers import PAGES, default_workspace
from app.ui.pages import (
    dataset_page,
    fairness_page,
    inspect_page,
    model_page,
    overview_page,
)

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
if st.session_state.get("navigation") not in PAGES:
    st.session_state["navigation"] = "Overview"
page = st.radio(
    "Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="navigation"
)
workspace = st.session_state["workspace"]
if page == "Overview":
    overview_page(workspace)
elif page == "Inspect an image":
    inspect_page(workspace)
elif page == "Dataset & evaluation":
    dataset_page(workspace)
elif page == "Research limitations":
    fairness_page(workspace)
else:
    model_page(workspace, root)
st.markdown(
    '<div class="footer">Academic research prototype · not clinically validated and not a substitute for professional medical evaluation.</div>',
    unsafe_allow_html=True,
)
