"""Streamlit app for live analysis

Usage
-------

How to start app:
run in terminal:
    streamlit run streamlit_apps\main.py --server.maxUploadSize 10000

The last number is the maximally allowed size for uploading raw_data in MBy
"""
import os
import time

os.environ['PREFECT__LOGGING__LEVEL'] = "ERROR"
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

import streamlit as st
from rydanalysis.auxiliary.streamlit_utils.st_state_patch import get as get_session_state
from rydanalysis.sequence_analysis import LiveAnalysis
from streamlit_apps.import_export import ImportExport, page_import_export


def page_set_analysis_parameters(importer=ImportExport(), analyzer=LiveAnalysis()):
    analyzer.streamlit_update_params()


def main():
    # state = get_session_state(
    #     import_export=ImportExport(),
    #     analyzer=LiveAnalysis()
    # )

    @st.cache(allow_output_mutation=True)
    def get_importer():
        return ImportExport()

    importer = get_importer()

    @st.cache(allow_output_mutation=True)
    def get_analyzer():
        return LiveAnalysis()

    analyzer = get_analyzer()

    pages = {
        "Import and export data": page_import_export,
        "Analysis settings": page_set_analysis_parameters
    }

    st.sidebar.title("Rydanalysis")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](importer, analyzer)

    time.sleep(1)
    st.experimental_rerun()


if __name__ == "__main__":
    main()