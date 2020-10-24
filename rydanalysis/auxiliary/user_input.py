import time

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


def choose_from_options(interface="notebook", message="", default="yes", options=None):
    if options is None:
        options = ["yes", "no"]
    if interface == "streamlit":
        st.text(message)
        cols = st.beta_columns(len(options))
        for col, option in zip(cols, options):
            if col.button(option):
                print("clicked")
                return option
        print("no button clicked")
        st.stop()
        print("hello")
    else:
        message = message + "Options: " + ", ".join(options)
        user_input(interface, message, default)
