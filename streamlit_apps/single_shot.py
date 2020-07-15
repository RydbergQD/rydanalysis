import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from rydanalysis.data_structure.ryd_data import *


def page_single_shot(state):
    st.write(state.sequence_selector)
    data = state.data
    if not data:
        return None

    variables = data.ryd_data.variables
    tmstp = st.selectbox("Choose time:", variables.index.values, )
    shot = data.sel(tmstp=tmstp)
    shot = shot.squeeze()

    if shot.ryd_data.has_traces:
        plot_ion_trace(shot)

    if shot.ryd_data.has_images:
        plot_atoms(shot)


def plot_ion_trace(shot):
    st.markdown('## Ion trace')
    peak_height = st.sidebar.number_input('minimal peak height', min_value=0., value=0.02)
    peak_width = st.sidebar.number_input('minimal peak width [in ns]', min_value=0., value=1.)
    trace = -shot.scope_traces
    peaks = trace.peaks.find_peaks(height=peak_height, width=peak_width * 1e-9)

    if st.checkbox('Show ion trace', value=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trace.time, y=trace.values, mode='lines', name='ion trace'))
        fig.add_trace(go.Scatter(x=peaks.time, y=peaks.values, mode='markers', name='peaks'))
        st.plotly_chart(fig, use_container_width=True)


def plot_images(shot):
    st.markdown('## Images')

    st.multiselect('What do you want to plot?',
                   ['background image', 'atom image', 'light_image', 'transmission', 'optical depth', 'density'],
                   ['background image', 'atom image', 'light_image', 'transmission'])
    plot_background(shot)


def plot_atoms(shot: xr.Dataset):
    atoms = shot['image_03']
    fig = px.imshow(atoms, aspect='equal')
    st.plotly_chart(fig)


def plot_background(shot: xr.Dataset):
    background = shot['image_05']
    fig = go.Figure(
        data=go.Heatmap(
            z=background,
            x=background.y,
            y=background.x,
            hoverongaps=False),
        layout=go.Layout(
            xaxis=dict(scaleanchor="y", scaleratio=1,),
        )
    )
    st.plotly_chart(fig)
