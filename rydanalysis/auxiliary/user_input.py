from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import streamlit as st

from .streamlit_utils.progress_bar import streamlit_tqdm


def custom_tqdm(iterable, interface="notebook", desc="", leave=True):
    if interface == "console":
        return tqdm(iterable, desc=desc, leave=leave)
    elif interface == "notebook":
        return tqdm_notebook(iterable, desc=desc, leave=leave)
    elif interface == "streamlit":
        return streamlit_tqdm(iterable, desc=desc, leave=leave)
    else:
        return iterable


def user_input(interface="notebook", message="", default=""):
    if interface == "streamlit":
        return st.text_input(message, value=default)
    else:
        return input(message)
