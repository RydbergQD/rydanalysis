import xarray as xr
from .ion_seqeunce_analysis import get_ion_summary
from .image_sequence_analysis import ImageParameters


class LiveAnalysis:
    def __init__(self, old_structure, image_parameters=ImageParameters()):
        self.old_structure = old_structure
        self.image_parameters = image_parameters
        new_data = self.initialize_raw_data()
        self.raw_data = new_data.copy()

        self.fit_ds, self.analysis_summary = self.initialize_summary(new_data)

    def initialize_raw_data(self):
        tmstps = self.old_structure.extract_tmstps()
        tmstp_batch = tmstps[:self.old_structure.batch_size]
        return self.old_structure.get_raw_data(tmstp_batch)

    def initialize_summary(self, raw_data):
        ion_summary = get_ion_summary(raw_data)
        fit_ds, image_summary = self.fit_images(raw_data)
        return fit_ds, xr.merge([ion_summary, image_summary])

    def update(self):
        new_data = self.update_raw_data()
        new_ion_summary = get_ion_summary(new_data)
        self.analysis_summary = xr.merge([self.analysis_summary, new_ion_summary])

    def update_raw_data(self):
        tmstps = self.old_structure.extract_tmstps()
        old_tmstps = self.raw_data.tmstp.values
        tmstp_batch = [t for t in tmstps if t not in old_tmstps][:self.old_structure.batch_size]
        new_raw_data = self.old_structure.get_raw_data(tmstp_batch)
        self.raw_data = xr.merge([self.raw_data, new_raw_data])
        return new_raw_data

    def fit_images(self, data):
        t_exp = data.parameters.sel(param_dim="tCAM", drop=True) * 1e3
        images = data.drop_vars("parameters").drop_dims("param_dim")
        return self.image_parameters.analyse_images(images, t_exp)
