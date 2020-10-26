import streamlit as st
from pathlib import Path
import datetime
from dataclasses import dataclass, field
from typing import Any

from rydanalysis.data_structure.extract_old_structure import OldStructure, load_data
from rydanalysis.sequence_analysis import LiveAnalysis


def get_default_path():
    base_path = Path(r"\\147.142.18.81\rydberg\data")
    date = datetime.date.today()
    strf_date = date.strftime('%Y_%m_%d')
    scan_name = " "
    return str(base_path / strf_date / scan_name)


def page_import_export(old_structure: OldStructure, raw_data: None, analyzer: LiveAnalysis):
    if raw_data is None:
        raw_data = []

    st.markdown("### Set options")
    st.file_uploader("load import/export settings. ")
    expander = st.beta_expander("Set import options", expanded=True)
    with expander:
        streamlit_set_path(old_structure)

    expander = st.beta_expander("Set export options", expanded=True)
    with expander:
        export_path = streamlit_set_export_path(old_structure)
    st.markdown("### Save and load data")
    col_save, col_load = st.beta_columns(2)
    with col_save:
        streamlit_save(old_structure, export_path, old_structure.append_to_old_data)
    with col_load:
        streamlit_load_hdf5(raw_data, export_path)

    if not raw_data:
        st.stop()
    if st.button("Analyze data"):
        analyzer.analyze_data(raw_data[0])


def streamlit_set_path(old_structure: OldStructure):
    path = st.text_input('Enter base path', value=str(old_structure.base_path))
    old_structure.base_path = Path(path)
    if not old_structure.base_path.is_dir():
        st.text("'Base _path is not a valid directory. '")

    # Choose date
    if st.checkbox("Choose from date?", True):
        date = st.date_input('Choose data', old_structure.date)
        old_structure.set_date(date)

    # Choose scan name
    if st.checkbox("Choose scan name", True):
        scan_names = old_structure.scan_names
        scan_name_index = old_structure.scan_name_index

        scan_name = st.selectbox('Choose run', scan_names, index=scan_name_index)
        old_structure.scan_name = scan_name


def streamlit_set_export_path(old_structure):
    old_structure.export_path = st.text_input(
        "Save as netcdf here", value=old_structure.export_path)
    export_path = Path(old_structure.export_path)

    # Add date to _path
    if old_structure.date is not None:
        old_structure.date_to_destiny = st.checkbox(
            "Add date to destiny _path:", old_structure.date_to_destiny)
    else:
        old_structure.date_to_destiny = False
    if old_structure.date_to_destiny:
        export_path = export_path / old_structure.strf_date / old_structure.scan_name

    st.text("""Data will be saved here: {}""".format(str(export_path)))
    return export_path


def streamlit_save(old_structure, export_path, append):
    old_structure.append_to_old_data = st.checkbox(
        "append to old data", old_structure.append_to_old_data)
    if st.button('save to hdf5'):
        (export_path / 'Analysis').mkdir(parents=True, exist_ok=True)
        old_structure.copy_sequences_variables(export_path)
        old_structure.save_data(export_path / "raw_data", append)


def streamlit_load_hdf5(raw_data, export_path):
    load_lazy = st.checkbox("Load lazy: ", value=False)
    if st.button('load_netcdf'):
        raw_data[0] = load_data(Path(export_path) / "raw_data", lazy=load_lazy)
