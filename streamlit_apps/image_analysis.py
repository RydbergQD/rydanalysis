import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from lmfit.models import GaussianModel

from rydanalysis.data_structure.ryd_data import *
import rydanalysis as ra
from dataclasses import dataclass
from typing import List


@dataclass()
class ImageParameters:
    light_name_index: int = 3
    atom_name_index: int = 1
    background_name_index: int = 5
    mask: List = None
    slice_option: str = "maximum"
    slice_position_x: float = None
    slice_position_y: float = None

    SLICE_OPTIONS = pd.Series(["maximum", "central moment", "manual"])

    @property
    def light_name(self) -> str:
        index = self.light_name_index
        return self.index_to_name(index)

    @property
    def atom_name(self) -> str:
        index = self.atom_name_index
        return self.index_to_name(index)

    @property
    def background_name(self) -> str:
        index = self.background_name_index
        return self.index_to_name(index)

    @staticmethod
    def index_to_name(index: int) -> str:
        return "image_" + str(index).zfill(2)

    @staticmethod
    def name_to_index(name: str) -> int:
        return int(name[-2:])

    def update_image_names(self, shot):
        st.sidebar.markdown("### Select images")
        light_name = st.sidebar.selectbox(
            "Select 'light image': ", options=shot.ryd_data.image_names,
            index=self.light_name_index)
        self.light_name_index = self.name_to_index(light_name)
        atom_name = st.sidebar.selectbox(
            "Select 'atom image': ", options=shot.ryd_data.image_names,
            index=self.atom_name_index)
        self.atom_name_index = self.name_to_index(atom_name)
        background_name = st.sidebar.selectbox(
            "Select 'background image': ", options=shot.ryd_data.image_names,
            index=self.background_name_index)
        self.background_name_index = self.name_to_index(background_name)

    def update_image_params(self, shot):
        st.sidebar.markdown("## Images")
        self.update_image_names(shot)

        st.sidebar.markdown("### Mask")
        self.mask = define_mask()

        st.sidebar.markdown("### Choose slices")
        default_index = self.SLICE_OPTIONS[self.SLICE_OPTIONS == self.slice_option].index[0]
        self.slice_option = st.sidebar.radio("How to choose slices:", options=self.SLICE_OPTIONS,
                                             index=int(default_index))
        if self.slice_option == "manual":
            self.slice_position_x = int(st.sidebar.text_input("x position", value="0"))
            self.slice_position_y = int(st.sidebar.text_input("y position", value="0"))


def analyse_images(shot: xr.Dataset, parameters: ImageParameters):
    if not shot.ryd_data.has_images:
        return None, None
    tmstp = shot.attrs["tmstp"]
    imaging = absorption_imaging(shot, parameters)
    image = imaging.density
    image_masked = apply_mask(image, parameters)

    center = find_center(image, parameters)
    fit_x = fit_slice(image_masked, center, 'x')
    fit_y = fit_slice(image_masked, center, 'y')
    fit_2d = fit_2d_gaussian(image_masked, fit_x, fit_y)

    fits = {"2d_": fit_2d, "slice_x_": fit_x, "slice_y_": fit_y}
    fit_ds = ra.merge_fits(fits)
    summary = summarize_fit(fit_ds, tmstp, center)
    return summary, fit_ds


def absorption_imaging(shot: xr.Dataset, parameters: ImageParameters, masked=False):
    background_name = parameters.background_name
    atom_name = parameters.atom_name
    light_name = parameters.light_name
    if masked:
        shot = apply_mask(shot, parameters)
    return ra.AbsorptionImaging.for_live_analysis(
        shot, background_name=background_name, atom_name=atom_name, light_name=light_name
    )


def apply_mask(dataset_or_array, parameters: ImageParameters):
    mask_vertices = parameters.mask
    masked = dataset_or_array.polygon_mask.apply_mask(mask_vertices)
    return masked.dropna(dim='x', how='all').dropna(dim='y', how='all')


def define_mask():
    mask_option = st.sidebar.selectbox(
        "Choose mask:", ["No mask", "from previous run", "Enter new mask"])
    if mask_option == "from previous run":
        mask_loader = st.sidebar.file_uploader("Load mask", type='csv')
        if mask_loader is not None:
            return pd.read_csv(mask_loader, index_col=False, header=None)
    elif mask_option == "Enter new mask":
        st.sidebar.text("Enter mask:")
        x = st.sidebar.text_input("X coordinates", "", )
        x = np.fromstring(x, sep=',')
        y = st.sidebar.text_input("Y coordinates", "", )
        y = np.fromstring(y, sep=',')
        if len(x) != len(y):
            st.sidebar.text("x and y need same length.")
            st.stop()
        elif len(x) == 0:
            st.stop()
        return np.array([x, y]).T


def find_center(image: xr.DataArray, parameters: ImageParameters):
    slice_option = parameters.slice_option
    mask_vertices = parameters.mask

    if slice_option == "maximum":
        return find_maximum(image, mask_vertices)
    elif slice_option == "manual":
        position_x = parameters.slice_position_x
        position_y = parameters.slice_position_y
        return image.sel(x=position_x, y=position_y, method='nearest')
    else:
        raise NotImplementedError("This method is not yet implemented :(")


def find_maximum(image, mask_vertices):
    filtered = gaussian_filter(image, sigma=2, mode='nearest')
    filtered = xr.DataArray(filtered, coords=image.coords)
    image_masked = image.polygon_mask.apply_mask(mask_vertices)

    maxima = peak_local_max(filtered.values)
    maximum = max([image_masked[tuple(m)] for m in maxima])
    return maximum


def fit_slice(image: xr.DataArray, center, variable='x'):
    fixed = 'y' if variable == 'x' else 'x'
    slice_x = image.sel({fixed: center[fixed].values}, drop=True, method='nearest')
    slice_x = slice_x.dropna(dim=variable)
    model = GaussianModel()

    params = model.guess(gaussian_filter(slice_x.values, sigma=1), x=slice_x[variable].values)
    return model.fit(slice_x.values, x=slice_x[variable], params=params, nan_policy='omit')


def fit_2d_gaussian(image, fit_x, fit_y):
    model = ra.Gaussian2D()
    amp = np.mean([fit_x.best_values['amplitude'], fit_y.best_values['amplitude']])
    cen_x = fit_x.best_values['center']
    cen_y = fit_y.best_values['center']
    sig_x = fit_x.best_values['sigma']
    sig_y = fit_y.best_values['sigma']

    params = model.make_params(amp=amp, cen_x=cen_x, cen_y=cen_y, sig_x=sig_x, sig_y=sig_y)
    return model.fit(image, params=params)


def summarize_fit(fit_ds, tmstp, center):
    summary = fit_ds.to_array().sel(fit='value', drop=True).to_series()
    summary["center_x"] = center.x.values
    summary["center_y"] = center.y.values
    summary.name = tmstp
    return summary


def plot_images(shot: xr.Dataset, parameters: ImageParameters, summary: pd.Series,
                fit_ds: xr.Dataset):
    st.markdown('# Images')
    plot_raw_images(shot, parameters)

    imaging = absorption_imaging(shot, parameters, masked=True)
    plot_live_analysis_images(imaging, summary, fit_ds)
    plot_slices(imaging.density, summary, fit_ds)


def plot_raw_images(shot: xr.Dataset, parameters: ImageParameters):
    background_name = parameters.background_name
    background = shot[background_name]
    atom_name = parameters.atom_name
    atom = shot[atom_name]
    light_name = parameters.light_name
    light = shot[light_name]
    st.markdown('## Raw images')

    image_dict = {'light image': light, 'atom image': atom, 'background image': background}

    image_names = st.multiselect(
        "Select plots to show:", options=list(image_dict.keys()), default=list(image_dict.keys())
    )

    n_images = len(image_names)
    heatmap_args = dict(hoverongaps=False, coloraxis="coloraxis")
    fig = make_subplots(
        rows=n_images, cols=1, shared_xaxes=True, shared_yaxes=True, row_heights=n_images * [800],
        subplot_titles=image_names, vertical_spacing=0.1
    )

    for i, name in enumerate(image_names):
        image = image_dict[name]
        image.plotly_image.add_trace(fig, row=i + 1, **heatmap_args)
        fig.update_yaxes(title_text=r"Y [μm]", row=i + 1, col=1)

    fig.update_xaxes(title_text=r"X [μm]", row=3, col=1)  # r'$X \text{[}\mu\text{m]}$'
    fig.update_layout(height=600)

    st.plotly_chart(fig)


def plot_live_analysis_images(imaging, summary: pd.Series, fit_ds: xr.Dataset):
    st.markdown('## Analysed images')

    all_image_names = ['density', 'transmission', 'optical depth', 'power', 'intensity',
                       'field strength', 'rabi frequency', 'saturation parameter']

    image_names = st.multiselect('What do you want to plot?', all_image_names, ['density'])
    n_images = len(image_names)

    vertical_spacing = 0.1
    fig = make_subplots(
        rows=n_images, cols=1, shared_xaxes=True, shared_yaxes=True, row_heights=n_images * [3000],
        subplot_titles=image_names, vertical_spacing=vertical_spacing
    )
    fig_slot = st.empty()

    image_args = dict(hoverongaps=False)
    for i, name in enumerate(image_names):
        update_image_args(image_args, i, n_images, vertical_spacing)
        image = get_image(imaging, name)
        image.plotly_image.add_trace(fig, row=i + 1, **image_args)
        if name == 'density':
            # Extract fit from summary
            params = fit_ds.to_parameters(par_prefix="2d_")
            model = ra.Gaussian2D()
            density = model.eval(params, x=image.x, y=image.y)
            density.plotly_image.add_contour(
                fig=fig, row=i + 1, col=1, contours_coloring='lines',
                line=dict(dash="solid", width=3, color="black")
            )
            st.markdown("## Fit result")
            st.markdown('### ' + name)
            st.table(data=params.to_report_table())
        fig.add_trace(trace=go.Scatter(x=summary["center_x"], y=summary["center_y"],
                                       marker=dict(symbol='x', size=10, color='blue')),
                      row=i + 1, col=1)
        fig.update_yaxes(title_text=r"Y [μm]", row=i + 1, col=1)

    fig.update_xaxes(title_text=r"X [μm]", row=3, col=1)
    fig.update_layout(height=n_images * 300 + 100)

    fig_slot.plotly_chart(fig)


def get_image(imaging, name: str):
    name = name.replace(' ', '_')
    return getattr(imaging, name)


def update_image_args(image_args, i, n_images, vertical_spacing):
    if n_images == 1:
        y_pos = 0.5
    else:
        y_pos = 1 - (i + 1) / 2 * vertical_spacing - (i + 0.3) / n_images
    image_args.update(colorbar=dict(len=1 / n_images - vertical_spacing, y=y_pos))


def plot_slices(image: xr.DataArray, summary: pd.Series, fit_ds: xr.Dataset):
    if not st.checkbox("Show slices?", value=True):
        return None

    fig = make_subplots(rows=2)
    plot_slice(fig, image, summary, fit_ds, variable="x", row=1)
    plot_slice(fig, image, summary, fit_ds, variable="y", row=2)
    st.plotly_chart(fig)


def plot_slice(fig, image, summary, fit_ds, variable='x', row=1):
    fixed = "y" if variable == "x" else "x"
    position = summary["center_" + fixed]
    image_slice = image.sel({fixed: position}, method='nearest', drop=True)
    x_values = image_slice[variable]

    params = fit_ds.to_parameters(par_prefix="slice_" + variable + "_")
    model = GaussianModel()
    fit = model.eval(params, x=x_values)

    fig.add_trace(
        go.Scatter(x=x_values, y=image_slice),
        row=row, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_values, y=fit, mode='lines'),
        row=row, col=1
    )
