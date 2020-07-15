from distutils.dir_util import copy_tree

import streamlit as st
from pathlib import Path
import datetime

from rydanalysis.data_structure.old_structure import OldStructure
from rydanalysis.data_structure.ryd_data import load_ryd_data


def page_import_export(state):

    st.markdown(
        r"""
        # Import and export data
        ## Import data
        
        Typical folders are_
        data in rydberg container: \\147.142.16.121\rydberg-share\rydberg\data
        data on axion: 
        example dataset on Titus laptop: C:\Users\titus\Dropbox\Rydberg\Small projects\test_rydanalysis
        """)
    selector_options = {
        'by path': 0,
        'by date': 1,
        'live': 2,
        'hdf5': 3
    }
    sequence_selector = st.radio('Choose how to select sequence:',
                                 ('by path', 'by date', 'live', 'hdf5', 'example_images'),
                                 index=selector_options[state.sequence_selector])
    state.sequence_selector = sequence_selector

    if sequence_selector == 'by date':
        load_by_date(state)
        export_by_date(state)
    elif sequence_selector == 'by path':
        load_by_path(state)
        export_by_path(state)
    elif sequence_selector == 'live':
        st.write('Not yet implemented.')
    elif sequence_selector == 'example_images':
        state.data = load_ryd_data(r"C:\Users\titus\Dropbox\Rydberg\Small "
                                   r"projects\test_rydanalysis\new_structure\2020_07_06\05_SNR_EIT_AT_pABSx0-8_pBlue3"
                                   r"-6\raw_data.h5")
    else:
        file = st.file_uploader('choose_file')
        if file:
            state.data = load_ryd_data(file)


def load_old_structure(state, path):
    if st.button('load'):
        old_structure = OldStructure(path)
        state.old_structure = old_structure
        state.data = old_structure.data


def load_by_date(state):
    # Choose base path
    ryd_container_path = r"\\147.142.16.121\rydberg-share\rydberg\data"
    base_path = Path(st.text_input('Enter base path', value=None))
    if not base_path.is_dir():
        return None

    # Choose date
    state.date = st.date_input('Choose data', datetime.date.today())
    state.strf_date = state.date.strftime('%Y_%m_%d')

    # Continue if date exists
    path = base_path / state.strf_date
    if path.is_dir():

        # load sequence
        state.scan_name = st.selectbox('Choose run', [x.name for x in path.iterdir() if x.is_dir()])
        state.origin_path = path / state.scan_name
        load_old_structure(state, state.origin_path)
    else:
        st.write('No experimental run was found on that date. Consider Changing the base path or verify the date.')


def load_by_path(state):
    path = Path(st.text_input('Enter path:'))
    state.origin_path = path
    state.scan_name = path.parts[-1]
    try:
        strf_date = path.parts[-2]
        state.date = datetime.datetime.strptime(strf_date, '%Y_%m_%d').date()
        state.strf_date = strf_date
    except ValueError:
        state.date = None
        state.strf_date = None
    load_old_structure(state, path)


def export_by_date(state):
    st.write("""## Export data""")
    path = Path(st.text_input("Save as netcdf here"))
    data = state.data
    origin_path = state.origin_path
    destiny_path = path / state.strf_date / state.scan_name
    export_data(data, origin_path, destiny_path)


def export_by_path(state):
    st.write("""## Export data""")
    data = state.data
    origin_path = state.origin_path

    path = Path(st.text_input("Save as netcdf here"))
    if state.strf_date:
        destiny_path = path / state.strf_date / state.scan_name
    else:
        destiny_path = path / state.scan_name

    export_data(data, origin_path, destiny_path)


def export_data(data, origin_path, destiny_path):
    st.text("""Data will be saved here:
    {}""".format(str(destiny_path)))

    if st.button('to_netcdf'):

        # Create directory where Analysis will be saved
        (destiny_path / 'Analysis').mkdir(parents=True)

        # Copy experimental sequences and variables to new location
        for dir_name in ('Experimental Sequences', 'Variables'):
            (destiny_path / dir_name).mkdir()
            copy_tree(
                str(origin_path / dir_name),
                str(destiny_path / dir_name)
            )

        # Save data as netcdf
        data.ryd_data.to_netcdf(destiny_path / 'raw_data.h5')