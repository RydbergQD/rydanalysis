import xarray as xr
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from rydanalysis.data_analysis.ions_analysis import PeaksSummaryAccessor, summarize_peak_description


@dataclass
class IonSequenceAnalysis:
    height: float = 0.02
    prominence: float = 0.
    threshold: float = 0.
    distance: float = 0.
    width: float = 0.3

    def peak_description(self, traces: xr.DataArray):
        return traces.peaks_summary.get_peak_description(
            self.height, self.prominence, self.threshold, self.distance, self.width
        )

    def full_analysis(self, traces: xr.DataArray):
        description = self.peak_description(traces)
        summary = summarize_peak_description(description)
        summary["ions_int"] = traces.peaks_summary.integrate(height=self.height)
        return description, summary

    def streamlit_update_params(self):
        st.markdown("## Peak finding settings:")
        self.height = st.number_input(
            'minimal peak height', min_value=0.005, value=self.height, step=0.0001, format="%g")
        self.width = st.number_input(
            'minimal peak width [in ns]', min_value=0., value=self.width)
        self.prominence = st.number_input(
            'minimal peak prominence', min_value=0., value=self.prominence,
            step=0.0001, format="%g")
        self.threshold = st.number_input(
            'minimal peak threshold', min_value=0., value=self.threshold,
            step=0.0001, format="%g")
        self.distance = st.number_input(
            'minimal peak distance [ns]', min_value=0., value=self.distance)

    def live_ion_analysis(self, shot):
        if not shot.ryd_data.has_traces:
            return None, None
        trace = -shot.scope_traces
        peaks = self.find_peaks(trace)
        return trace, peaks

    def find_peaks(self, trace: xr.DataArray):
        peak_height = self.height
        peak_width = self.width
        return trace.peaks.find_peaks(height=peak_height, width=peak_width * 1e-9)
