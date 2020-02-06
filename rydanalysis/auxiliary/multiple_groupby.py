import pandas as pd
import xarray as xr


def pandas_to_raw_data(pandas_df):
    da = xr.DataArray(pandas_df, dims=['shot', 'data_coords'])
    return da.unstack('data_coords')


class DataGroupby:
    def __init__(self, groupby):
        self._groupby = groupby

    @classmethod
    def wrapper(cls, func):
        def wrapped_func(self, *args, **kwargs):
            pandas_df = getattr(self._groupby, func)(*args, **kwargs)
            return pandas_to_raw_data(pandas_df)

        return wrapped_func

    def __iter__(self):
        for name, group in self._groupby:
            yield name, pandas_to_raw_data(group)

    def map(self, func, args=(), **kwargs):

        def _func(name, arr):
            result = func(arr, *args, **kwargs)
            if 'shot' not in result.dims:
                result = result.expand_dims('shot')
                return result.assign_coords(
                    coords={'shot': pd.MultiIndex.from_tuples([name], names=self._groupby.keys)})
            else:
                return result

        results = [_func(name, arr) for name, arr in self]
        return xr.concat(results, dim='shot')

    def reduce(
            self, func, dim=None, axis=None, keep_attrs=None, **kwargs
    ):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : `...`, str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """

        def reduce_array(ar):
            return ar.reduce(func, dim, axis, keep_attrs=keep_attrs, **kwargs)

        return self.map(reduce_array)


def inject_method(cls, func):
    setattr(DataGroupby, func, cls.wrapper(func))


NAN_REDUCE_METHODS = [
    "argmax",
    "argmin",
    "max",
    "min",
    "mean",
    "prod",
    "sum",
    "std",
    "var",
    "median",
]

for method in NAN_REDUCE_METHODS:
    inject_method(DataGroupby, method)


@xr.register_dataarray_accessor("multiple_groupby")
class MultipleGroupBy:
    """
    Wraps the pandas groupby functionality to xarray. Works currently only
     for images with the coordinates ['x', 'y', 'shot']
    """

    def __init__(self, arr):
        # Create pandas DataFrame with MultiColumns containing x and y and multiindex containing the shots multiindex
        data_coords = [dim for dim in arr.dims if dim != 'shot']
        print(data_coords)
        df = arr.stack({'data_coords': data_coords}).to_pandas()
        self._dataframe = df

    def __call__(self, group):
        gb = self._dataframe.groupby(group)
        return DataGroupby(gb)
