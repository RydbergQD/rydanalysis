import streamlit as st

from rydanalysis.data_structure.ryd_data import *
from .ryd_state import RydState
from .variables_explorer import plot_variables
from .image_analysis import analyse_images, plot_images
from .ion_analysis import live_ion_analysis, plot_ion_trace


def page_single_shot(state: RydState):
    st.write(state.sequence_selector)
    data = state.data
    if not data:
        return None

    variables: pd.DataFrame = data.ryd_data.variables
    tmstp = choose_tmstp(variables)
    shot = data.ryd_data.choose_shot(tmstp=tmstp)

    plot_variables(data.ryd_data.variables, tmstp)

    if shot.ryd_data.has_traces:
        if st.sidebar.checkbox("Change live analysis parameters: "):
            state.ion_parameters.update_ion_params()

        trace, peaks = live_ion_analysis(shot, state.ion_parameters)
        plot_ion_trace(trace, peaks)

    if shot.ryd_data.has_images:
        if st.sidebar.checkbox("Change live analysis parameters: "):
            state.image_parameters.update_image_params(shot)
        summary, fit_ds = analyse_images(shot, state.image_parameters)
        plot_images(shot, state.image_parameters, summary, fit_ds)


def choose_tmstp(variables):
    choose_option = st.radio(
        "Choose scan: ",
        options=["Choose from variables.", "Choose from timestamp.", "Choose last scan."],
        index=2
    )
    if choose_option == "Choose from timestamp.":
        tmstp = st.selectbox("Choose time:", variables.index.values, )
    elif choose_option == "Choose from variables.":
        for var_name, var in variables.T.iterrows():
            values = var.unique()
            val = st.selectbox(label=var_name, options=values)
            variables = variables.where(var == val).dropna()

        tmstp = st.selectbox("Choose time:", variables.index.values, )
    else:
        tmstps = variables.index.values
        tmstp = tmstps.max()
    return tmstp
