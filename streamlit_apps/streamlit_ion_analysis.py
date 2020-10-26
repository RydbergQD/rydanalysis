import xarray as xr
import streamlit as st
import plotly.graph_objects as go


def plot_ion_trace(trace, peaks):
    st.markdown('## Ion trace')

    if st.checkbox('Show ion trace', value=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trace.time, y=trace.values, mode='lines', name='ion trace'))
        fig.add_trace(go.Scatter(x=peaks.time, y=peaks.values, mode='markers', name='peaks'))
        st.plotly_chart(fig, use_container_width=True)
