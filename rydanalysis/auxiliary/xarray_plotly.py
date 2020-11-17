import xarray as xr
import plotly.graph_objects as go


@xr.register_dataarray_accessor("plotly_image")
class XarrayPlotlyImage:
    def __init__(self, image):
        self.image = image

    def add_trace(self, fig: go.Figure, row: int = 1, col: int = 1, **heatmapargs):
        image = self.image
        fig.add_trace(
            go.Heatmap(z=image.T, x=image.x, y=image.y, **heatmapargs), row=row, col=col
        )

        if len(fig.data) > 1:
            # fig.update_xaxes(matches='x', constrain="domain", row=row, col=col)
            fig.update_yaxes(matches="y", constrain="domain", row=row, col=col)

    def add_contour(self, fig: go.Figure, row: int = 1, col: int = 1, **contourargs):
        image = self.image
        fig.add_trace(
            go.Contour(z=image.T, x=image.x, y=image.y, **contourargs), row=row, col=col
        )

        if len(fig.data) > 1:
            # fig.update_xaxes(matches='x', constrain="domain", row=row, col=col)
            fig.update_yaxes(matches="y", constrain="domain", row=row, col=col)
