import streamlit as st
import pandas as pd
from io import StringIO

from streamlit_apps.st_state_patch import State

from rydanalysis.auxiliary.masks import PolygonMask


def define_mask(state):
    st.sidebar.subheader('Mask')

    choose_mask = st.sidebar.selectbox("Choose mask:", ["No mask", "from previous run", "Enter new mask"], )
    if choose_mask == "No mask":
        state.mask_contour = None
    elif choose_mask == "from previous run":
        mask_loader = st.sidebar.file_uploader("Load mask", type='csv')
        if mask_loader is not None:
            state.mask_contour = pd.read_csv(mask_loader, index_col=False, header=None)
            mask_loader.close()
    else:
        text = st.sidebar.text_area("Enter new mask", "", height=2)
        if text != "":
            state.mask_contour = pd.read_csv(StringIO(text), index_col=False, header=None)

    if state.mask_contour is not None:
        st.sidebar.dataframe(state.mask)



if __name__ == '__main__':
    state = State(key="user metadata")
    if not state:
        state.mask = None
    define_mask(state)
