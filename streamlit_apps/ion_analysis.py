import xarray as xr
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass


@dataclass
class IonParameters:
    peak_height: float = 0.02
    peak_width: float = 3.

    def update_ion_params(self):
        st.markdown('## Ion trace')
        st.sidebar.markdown("## Peak finding options:")
        self.peak_height = st.sidebar.number_input(
            'minimal peak height', min_value=0., value=self.peak_height)
        self.peak_width = st.sidebar.number_input(
            'minimal peak width [in ns]', min_value=0., value=self.peak_width)


def live_ion_analysis(shot, parameters: IonParameters):
    if not shot.ryd_data.has_traces:
        return None, None
    trace = -shot.scope_traces
    peaks = find_peaks(trace, parameters)
    return trace, peaks


def find_peaks(trace: xr.DataArray, parameters: IonParameters):
    peak_height = parameters.peak_height
    peak_width = parameters.peak_width
    return trace.peaks.find_peaks(height=peak_height, width=peak_width * 1e-9)


def plot_ion_trace(trace, peaks):
    st.markdown('## Ion trace')

    if st.checkbox('Show ion trace', value=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trace.time, y=trace.values, mode='lines', name='ion trace'))
        fig.add_trace(go.Scatter(x=peaks.time, y=peaks.values, mode='markers', name='peaks'))
        st.plotly_chart(fig, use_container_width=True)
