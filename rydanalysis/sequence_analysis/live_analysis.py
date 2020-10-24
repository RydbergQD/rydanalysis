import xarray as xr
import pandas as pd
from .ion_sequence_analysis import IonSequenceAnalysis
from .image_sequence_analysis import ImageParameters


class LiveAnalysis:
    def __init__(self, raw_data=None, image_parameters=ImageParameters(),
                 ion_analysis=IonSequenceAnalysis()):
        self.image_parameters = image_parameters
        self.ion_analysis = ion_analysis
        if raw_data is not None:
            raw_data = raw_data.copy()
            self.ion_description, self.fit_ds, self.summary = self.analyze_data(raw_data)
        else:
            self.ion_description = None
            self.fit_ds = None
            self.summary = None

    def analyze_data(self, raw_data):
        ion_description, ion_summary = self.ion_analysis.full_analysis(raw_data.scope_traces)
        fit_ds, image_summary = self.fit_images(raw_data)
        return ion_description, fit_ds, xr.merge([ion_summary, image_summary])

    def update(self, raw_data):
        new_description, new_ds, new_summary = self.analyze_data(raw_data.copy())
        self.ion_description = pd.concat([self.ion_description, new_description], axis=0)
        self.fit_ds = xr.concat([self.fit_ds, new_ds], dim="shot")
        self.summary = xr.concat([self.summary, new_summary], dim="shot")

    def fit_images(self, data):
        t_exp = data.parameters.sel(param_dim="tCAM", drop=True) * 1e3
        images = data.drop_vars("parameters").drop_dims("param_dim")
        return self.image_parameters.analyse_images(images, t_exp)

    def streamlit_update_params(self):
        self.ion_analysis.streamlit_update_params()
