import pytest
from pathlib import Path

import rydanalysis as ra


def test_to_json(image_data):
    analysis_parameters = ra.AnalysisParameters.from_data(image_data)
    assert analysis_parameters.date == "2020_07_06"


def test_fit(image_data):
    shot = image_data.isel(shot=0, drop=True)
    imaging = ra.AbsorptionImaging.for_live_analysis(shot)
    data = imaging.density
    model = ra.Gaussian2D()
    params = model.guess(data, use_quantile=True)
    result = model.fit(data, params, nan_policy='omit')
    result