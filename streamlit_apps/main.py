"""Streamlit app for live analysis

Usage
-------

How to start app:
run in terminal:
    streamlit run streamlit_apps\main.py --server.maxUploadSize 10000

The last number is the maximally allowed size for uploading raw_data in MBy
"""
import os
from time import sleep

os.environ['PREFECT__LOGGING__LEVEL'] = "ERROR"
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

import streamlit as st
from streamlit_apps.st_state_patch import _get_state
from streamlit_apps.import_export import page_import_export
from streamlit_apps.page_single_shot import page_single_shot


import xarray as xr
import numpy as np

from prefect import Flow, task, Parameter


def main():
    state = _get_state()

    pages = {
        "Import and export data": page_import_export,
        "Single shot": page_single_shot,
    }

    st.sidebar.title(":floppy_disk: Rydanalysis")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    # pages[page](state)

    r, flow = build_flow()

    st.write(flow.visualize())

    if st.button("run prefect"):
        try_prefect(state, r, flow)
    st.slider('bla', 0, 10, 0)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


@task
def collect_fast_csv_kwargs(flow=Flow("collect_fast_csv_kwargs")):
    with flow:
        Parameter()


def build_flow():
    with Flow("xarray flow") as flow:
        arrays = build_array.map(np.arange(10))
        r = reduce(arrays)
    return r, flow


def try_prefect(state, r, flow):

    prefect_state = flow.run(executor=state.executor)

    array: xr.DataArray = prefect_state.result[r].result
    st.write(array)


@task
def build_array(i):
    sleep(0.1)
    return xr.DataArray(np.arange(10), dims=['a'], coords={'a': (1 + i) * np.arange(10)})


@task
def reduce(array_list):
    return xr.concat(array_list, dim='a')





if __name__ == "__main__":
    main()
