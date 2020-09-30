import warnings
from copy import deepcopy
import sys
from lmfit import Model, Parameter
from lmfit.model import _ensureMatplotlib, propagate_err, ModelResult, isnull, _align, CompositeModel
import xarray as xr
import inspect
import numpy as np
import operator
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_reducer(option):
    """Factory function to build a parser for complex numbers.

    Parameters
    ----------
    option : str
        Should be one of `['real', 'imag', 'abs', 'angle']`. Implements the
        numpy function of the same name.

    Returns
    -------
    callable
        See docstring for `reducer` below.

    """
    if option not in ['real', 'imag', 'abs', 'angle']:
        raise ValueError("Invalid parameter name ('%s') for function 'propagate_err'." % option)

    def reducer(array):
        """Convert a complex array to a real array.

        Several conversion methods are available and it does nothing to a
        purely real array.

        Parameters
        ----------
        array : array-like
            Input array. If complex, will be converted to real array via one
            of the following numpy functions: `real`, `imag`, `abs`, or `angle`.

        Returns
        -------
        numpy.array
            Returned array will be purely real.

        """
        if np.any(np.iscomplex(array)):
            parsed_array = getattr(np, option)(array)
        else:
            parsed_array = array

        return parsed_array

    return reducer


class Model2d(Model):
    stacked_name = 'stacked_args'

    def __init__(self, func, independent_vars=None, param_names=None,
                 nan_policy='raise', prefix='', name=None, **kws):
        Model.__init__(self, func, independent_vars=independent_vars, param_names=param_names,
                       nan_policy=nan_policy, prefix=prefix, name=name, **kws)

    def __add__(self, other):
        """+"""
        return CompositeModel2d(self, other, operator.add)

    def __sub__(self, other):
        """-"""
        return CompositeModel2d(self, other, operator.sub)

    def __mul__(self, other):
        """*"""
        return CompositeModel2d(self, other, operator.mul)

    def __div__(self, other):
        """/"""
        return CompositeModel2d(self, other, operator.truediv)

    def __truediv__(self, other):
        """/"""
        return CompositeModel2d(self, other, operator.truediv)

    def _parse_params(self):
        """Build parameters from function arguments."""
        if self.func is None:
            return
        # need to fetch the following from the function signature:
        #   pos_args: list of positional argument names
        #   kw_args: dict of keyword arguments with default values
        #   keywords_:  name of **kws argument or None
        # 1. limited support for asteval functions as the model functions:
        if hasattr(self.func, 'argnames') and hasattr(self.func, 'kwargs'):
            pos_args = self.func.argnames[:]
            keywords_ = None
            kw_args = {}
            for name, defval in self.func.kwargs:
                kw_args[name] = defval
        # 2. modern, best-practice approach: use inspect.signature
        elif sys.version_info > (3, 4):
            pos_args = []
            kw_args = {}
            keywords_ = None
            sig = inspect.signature(self.func)
            for fnam, fpar in sig.parameters.items():
                if fpar.kind == fpar.VAR_KEYWORD:
                    keywords_ = fnam
                elif fpar.kind == fpar.POSITIONAL_OR_KEYWORD:
                    if fpar.default == fpar.empty:
                        pos_args.append(fnam)
                    else:
                        kw_args[fnam] = fpar.default
                elif fpar.kind == fpar.VAR_POSITIONAL:
                    raise ValueError("varargs '*%s' is not supported" % fnam)
        # 3. Py2 compatible approach
        else:
            argspec = inspect.getargspec(self.func)
            keywords_ = argspec.keywords
            pos_args = argspec.args
            kw_args = {}
            if argspec.defaults is not None:
                for val in reversed(argspec.defaults):
                    kw_args[pos_args.pop()] = val
        # inspection done

        self._func_haskeywords = keywords_ is not None
        self._func_allargs = pos_args + list(kw_args.keys())
        allargs = self._func_allargs

        if len(allargs) == 0 and keywords_ is not None:
            return

        # default independent_var = 1st and 2nd argument
        if self.independent_vars is None:
            self.independent_vars = pos_args[:2]

        # default param names: all positional args
        # except independent variables
        self.def_vals = {}
        might_be_param = []
        if self._param_root_names is None:
            self._param_root_names = pos_args[:]
            for key, val in kw_args.items():
                if (not isinstance(val, bool) and
                        isinstance(val, (float, int))):
                    self._param_root_names.append(key)
                    self.def_vals[key] = val
                elif val is None:
                    might_be_param.append(key)
            for p in self.independent_vars:
                if p in self._param_root_names:
                    self._param_root_names.remove(p)

        new_opts = {}
        for opt, val in self.opts.items():
            if (opt in self._param_root_names or opt in might_be_param and
                    isinstance(val, Parameter)):
                self.set_param_hint(opt, value=val.value,
                                    min=val.min, max=val.max, expr=val.expr)
            elif opt in self._func_allargs:
                new_opts[opt] = val
        self.opts = new_opts

        names = []
        if self._prefix is None:
            self._prefix = ''
        for pname in self._param_root_names:
            names.append("%s%s" % (self._prefix, pname))

        # check variables names for validity
        # The implicit magic in fit() requires us to disallow some
        fname = self.func.__name__
        for arg in self.independent_vars:
            if arg not in allargs or arg in self._forbidden_args:
                raise ValueError(self._invalid_ivar % (arg, fname))
        for arg in names:
            if (self._strip_prefix(arg) not in allargs or
                    arg in self._forbidden_args):
                raise ValueError(self._invalid_par % (arg, fname))
        # the following as been changed from OrderedSet for the time being.
        self._param_names = names[:]

    def _update_kwargs(self, data, **kwargs):
        for var in self.independent_vars:
            if var in kwargs:
                continue
            elif var in data.coords:
                kwargs[var] = data[var]
            else:
                raise ValueError('Not initialized variable ', var)
        return kwargs

    def create_xarray(self, data, **kwargs):
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(data, dims=self.independent_vars)

        for var, var_data in kwargs.items():
            if var in self.independent_vars:
                var_data = kwargs[var]
                data = data.assign_coords(**{var: var_data})
        return data

    def stack(self, data):
        return data.stack({self.stacked_name: self.independent_vars})

    def unstack(self, data):
        return data.unstack(self.stacked_name)

    def _residual(self, params, data, weights, **kwargs):
        """Return the residual.

        Default residual: (data-model)*weights.

        If the model returns complex values, the residual is computed by
        treating the real and imaginary parts separately. In this case,
        if the weights provided are real, they are assumed to apply
        equally to the real and imaginary parts. If the weights are
        complex, the real part of the weights are applied to the real
        part of the residual and the imaginary part is treated
        correspondingly.

        Since the underlying scipy.optimize routines expect numpy.float
        arrays, the only complex type supported is np.complex.

        The "ravels" throughout are necessary to support pandas.Series.

        """
        model = self.eval(params, **kwargs)
        if self.nan_policy == 'raise' and not np.all(np.isfinite(model)):
            msg = ('The model function generated NaN values and the fit '
                   'aborted! Please check your model function and/or set '
                   'boundaries on parameters where applicable. In cases like '
                   'this, using "nan_policy=\'omit\'" will probably not work.')
            raise ValueError(msg)

        diff = model - data

        if diff.dtype == np.complex:
            raise NotImplementedError("Complex values not yet supported. Can be fixed by properly stacking the values.")
            # # data/model are complex
            # diff = diff.ravel().view(np.float)
            # if weights is not None:
            #     if weights.dtype == np.complex:
            #         # weights are complex
            #         weights = weights.ravel().view(np.float)
            #     else:
            #         # real weights but complex data
            #         weights = (weights + 1j * weights).ravel().view(np.float)
        if weights is not None:
            diff *= weights

        return self.stack(diff).dropna(self.stacked_name)

    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            nan_policy=None, calc_covar=True, **kwargs):
        """Fit the model to the data using the supplied Parameters.
        Parameters
        ----------
        data : array_like
            Array of data to be fit.
        params : Parameters, optional
            Parameters to use in fit (default is None).
        weights : array_like of same size as `data`, optional
            Weights to use for the calculation of the fit residual (default
            is None).
        method : str, optional
            Name of fitting method to use (default is `'leastsq'`).
        iter_cb : callable, optional
            Callback function to call at each iteration (default is None).
        scale_covar : bool, optional
            Whether to automatically scale the covariance matrix when
            calculating uncertainties (default is True).
        verbose: bool, optional
            Whether to print a message when a new parameter is added because
            of a hint (default is True).
        nan_policy : str, optional, one of 'raise' (default), 'propagate', or 'omit'.
            What to do when encountering NaNs when fitting Model.
        fit_kws: dict, optional
            Options to pass to the minimizer being used.
        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True) for
            solvers other than `leastsq` and `least_squares`. Requires the
            `numdifftools` package to be installed.
        **kwargs: optional
            Arguments to pass to the  model function, possibly overriding
            params.
        Returns
        -------
        ModelResult2d
        Examples
        --------
        Take `t` to be the independent variable and data to be the curve we
        will fit. Use keyword arguments to set initial guesses:
        >>> result = my_model.fit(data, tau=5, N=3, t=t)
        Or, for more control, pass a Parameters object.
        >>> result = my_model.fit(data, params, t=t)
        Keyword arguments override Parameters.
        >>> result = my_model.fit(data, params, tau=5, t=t)
        Notes
        -----
        1. if `params` is None, the values for all parameters are
        expected to be provided as keyword arguments.  If `params` is
        given, and a keyword argument for a parameter value is also given,
        the keyword argument will be used.
        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.
        3. Parameters (however passed in), are copied on input, so the
        original Parameter objects are unchanged, and the updated values
        are in the returned `ModelResult`.
        """
        if params is None:
            try:
                params = self.guess(data, **kwargs)
            except NotImplementedError:
                params = self.make_params(verbose=verbose)
        else:
            params = deepcopy(params)

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & set(self.param_names)
        for name in param_kwargs:
            p = kwargs[name]
            if isinstance(p, Parameter):
                p.name = name  # allows N=Parameter(value=5) with implicit name
                params[name] = deepcopy(p)
            else:
                params[name].set(value=p)
            del kwargs[name]

        # All remaining kwargs should correspond to independent variables.
        for name in kwargs:
            if name not in self.independent_vars:
                warnings.warn("The keyword argument %s does not " % name +
                              "match any arguments of the model function. " +
                              "It will be ignored.", UserWarning)

        # If any parameter is not initialized raise a more helpful error.
        missing_param = any([p not in params.keys()
                             for p in self.param_names])
        blank_param = any([(p.value is None and p.expr is None)
                           for p in params.values()])
        if missing_param or blank_param:
            msg = ('Assign each parameter an initial value by passing '
                   'Parameters or keyword arguments to fit.\n')
            missing = [p for p in self.param_names if p not in params.keys()]
            blank = [name for name, p in params.items()
                     if p.value is None and p.expr is None]
            msg += 'Missing parameters: %s\n' % str(missing)
            msg += 'Non initialized parameters: %s' % str(blank)
            raise ValueError(msg)

        # Create xarray from data and weights
        # convert other iterables to xarrays indeces.
        # if isinstance(data, xr.DataArray):
        #     data = data.transpose(*self.independent_vars)
        # else:
        #     data = xr.DataArray(data, dims=self.independent_vars)
        # if isinstance(weights, xr.DataArray):
        #     weights = weights.transpose(*self.independent_vars)
        # elif weights is not None:
        #     weights = xr.DataArray(weights, dims=self.independent_vars)
        #
        # for var in self.independent_vars:
        #     if var in kwargs:
        #         var_data = kwargs[var]
        #     elif var in data.coords:
        #         var_data = data.coords[var]
        #     else:
        #         raise ValueError('Not initialized variable ', var)
        #     data = data.assign_coords(**{var: var_data})
        #     kwargs[var] = data[var]
        #     if weights is not None:
        #         weights = weights.assign_coords(**{var: var_data})
        kwargs = self._update_kwargs(data, **kwargs)
        data = self.create_xarray(data, **kwargs)
        if weights is not None:
            weights = self.create_xarray(weights, **kwargs)

        # Handle null/missing values.
        if nan_policy is not None:
            self.nan_policy = nan_policy

        mask = None
        if self.nan_policy == 'omit':
            mask = ~data.isnull()
            if mask is not None:
                data = data.where(mask, drop=True)
            if weights is not None:
                weights = weights.where(mask, drop=True)
            mask = None

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.

        for var in self.independent_vars:
            if not np.isscalar(kwargs[var]):
                # print("Model fit align ind dep ", var, mask.sum())
                kwargs[var] = _align(kwargs[var], mask, data)

        if fit_kws is None:
            fit_kws = {}

        output = ModelResult2d(self, params, method=method, iter_cb=iter_cb,
                               scale_covar=scale_covar, fcn_kws=kwargs,
                               nan_policy=self.nan_policy, calc_covar=calc_covar,
                               **fit_kws)
        output.fit(data=data, weights=weights)
        output.components = self.components
        return output

    def eval(self, params=None, **kwargs):
        """Evaluate the model with supplied parameters and keyword arguments.

        Parameters
        -----------
        params : Parameters, optional
            Parameters to use in Model.
        **kwargs : optional
            Additional keyword arguments to pass to model function.

        Returns
        -------
        xr.DataArray
            Value of model given the parameters and other arguments.

        Notes
        -----
        1. if `params` is None, the values for all parameters are
        expected to be provided as keyword arguments.  If `params` is
        given, and a keyword argument for a parameter value is also given,
        the keyword argument will be used.

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        """
        for variable in self.independent_vars:
            data_var = kwargs[variable]
            kwargs[variable] = xr.DataArray(data_var, coords={variable: data_var}, dims=variable)

        return self.func(**self.make_funcargs(params, kwargs))

    def copy(self, **kwargs):
        """DOES NOT WORK."""
        raise NotImplementedError("Model.copy does not work. Make a new Model")


class CompositeModel2d(CompositeModel, Model2d):

    def copy(self, **kwargs):
        """DOES NOT WORK."""
        raise NotImplementedError("Model.copy does not work. Make a new Model")


class ModelResult2d(ModelResult):
    """Result from the Model fit.
    This has many attributes and methods for viewing and working with
    the results of a fit using Model. It inherits from Minimizer, so
    that it can be used to modify and re-run the fit for the Model.
    """

    def __init__(self, model, params, data=None, weights=None,
                 method='leastsq', fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, nan_policy='raise',
                 calc_covar=True, **fit_kws):
        super().__init__(model, params, data=data, weights=weights,
                         method=method, fcn_args=fcn_args, fcn_kws=fcn_kws,
                         iter_cb=iter_cb, scale_covar=scale_covar, nan_policy=nan_policy,
                         calc_covar=calc_covar, **fit_kws)

    @_ensureMatplotlib
    def plot_fit(self, ax=None, datafmt='o', fitfmt='-', initfmt='--',
                 fit_color='black', init_color='tab:grey',
                 xlabel=None, ylabel=None, yerr=None, numpoints=None,
                 data_kws=None, fit_kws=None, init_kws=None, ax_kws=None,
                 show_init=False, parse_complex='abs'):
        """Plot the fit results using matplotlib, if available.
        The plot will include the data points, the initial fit curve (optional,
        with `show_init=True`), and the best-fit curve. If the fit model
        included weights or if `yerr` is specified, errorbars will also be
        plotted.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : str, optional
            Matplotlib format string for data points. (Not used in 2d PLot)
        fitfmt : str, optional
            Matplotlib format string for fitted curve.
        initfmt : str, optional
            Matplotlib format string for initial conditions for the fit.
        fit_color : str, optional
            color of the fit contour line, default is black.
        init_color : str, optional
            color of the init contour lines, default is tab:grey.
        xlabel : str, optional
            Matplotlib format string for labeling the x-axis.
        ylabel : str, optional
            Matplotlib format string for labeling the y-axis.
        yerr : numpy.ndarray, optional
            Array of uncertainties for data array.
        numpoints : tuple, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total. First value for x, second for y.
        data_kws : dict, optional
            Keyword arguments passed on to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed on to the plot function for fitted curve.
        init_kws : dict, optional
            Keyword arguments passed on to the plot function for the initial
            conditions of the fit.
        ax_kws : dict, optional
            Keyword arguments for a new axis, if there is one being created.
        show_init : bool, optional
            Whether to show the initial conditions for the fit (default is False).
        parse_complex : str, optional
            How to reduce complex data for plotting.
            Options are one of `['real', 'imag', 'abs', 'angle']`, which
            correspond to the numpy functions of the same name (default is 'abs').
        Returns
        -------
        matplotlib.axes.Axes
        Notes
        -----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.
        If `yerr` is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If `yerr` is
        not specified and the fit includes weights, `yerr` set to 1/self.weights
        If model returns complex data, `yerr` is treated the same way that
        weights are in this case.
        If `ax` is None then `matplotlib.pyplot.gca(**ax_kws)` is called.
        See Also
        --------
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.
        """
        from matplotlib import pyplot as plt
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_kws is None:
            ax_kws = {}

        # The function reduce_complex will convert complex vectors into real vectors
        reduce_complex = get_reducer(parse_complex)

        independent_vars = self.model.independent_vars[:2]

        if not isinstance(ax, plt.Axes):
            ax = plt.gca(**ax_kws)

        xy_array_dense = self.build_dense_xarray(numpoints)

        if show_init:
            init_data = reduce_complex(self.model.eval(self.init_params, **xy_array_dense))
            init_data.plot.contour(colors=init_color, linestyles=initfmt, ax=ax, **init_kws)

        data = reduce_complex(self.data)
        data.plot(ax=ax, label='data', center=False, **data_kws)

        best_fit = reduce_complex(self.model.eval(self.params, **xy_array_dense))
        best_fit.plot.contour(colors=fit_color, linestyles=fitfmt, ax=ax, **fit_kws)

        ax.set_title(self.model.name)
        # if xlabel is None:
        #    ax.set_xlabel(independent_vars[0])
        # else:
        #    ax.set_xlabel(xlabel)
        # if ylabel is None:
        #    ax.set_ylabel(independent_vars[1])
        # else:
        #    ax.set_ylabel(ylabel)
        # ax.legend(loc='best')
        return ax

    def build_dense_xarray(self, numpoints=None):
        independent_vars = self.model.independent_vars[:2]

        xy_array_dense = {}
        for i, independent_var in enumerate(independent_vars):
            xy_array = self.userkws[independent_var]

            # make a dense array for x-axis if data is not dense
            if numpoints is not None and len(self.data) < numpoints[i]:
                xy_array_dense[independent_var] = np.linspace(min(xy_array), max(xy_array), numpoints)
            else:
                xy_array_dense[independent_var] = xy_array
        return xy_array_dense

    def plotly_fit(self, fig: go.Figure, row: int = 1, col: int = 1,
                   numpoints=None,
                   show_init=False, init_color='grey', initfmt='dash', init_width=3, init_kws=None,
                   fit_color='black', fitfmt='solid', fit_width=3, fit_kws=None,
                   data_kws=None,
                   parse_complex='abs', weighted=True, heatmap_args=None):
        # The function reduce_complex will convert complex vectors into real vectors
        if fit_kws is None:
            fit_kws = {}
        if data_kws is None:
            data_kws = {}
        if init_kws is None:
            init_kws = {}
        if heatmap_args is None:
            heatmap_args = {}

        reduce_complex = get_reducer(parse_complex)

        xy_array_dense = self.build_dense_xarray(numpoints)

        if show_init:
            init_data = reduce_complex(self.model.eval(self.init_params, **xy_array_dense))
            init_data.plotly_image.add_contour(
                fig=fig, row=row, col=col, contours_coloring='lines',
                line=dict(dash=initfmt, width=init_width, color=init_color),
                **init_kws
            )

        data = reduce_complex(self.data)
        data.plotly_image.add_trace(fig=fig, row=row, col=col, name='data', **data_kws)

        best_fit = reduce_complex(self.model.eval(self.params, **xy_array_dense))

        best_fit.plotly_image.add_contour(
            fig=fig, row=row, col=col, contours_coloring='lines',
            line=dict(dash=fitfmt, width=fit_width, color=fit_color),
            **fit_kws
        )
        return fig

    def plotly_plot(self, ):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('residuals', 'fit'),
                            row_heights=[0.3, 0.7])
        self.plotly_residuals(fig, row=1, col=1)
        self.plot_fit(fig, row=2, col=1)
        fig.update_yaxes(yaxis='y', row=2, col=1)

    @_ensureMatplotlib
    def plot_residuals(self, ax=None, datafmt='o', yerr=None, data_kws=None,
                       fit_kws=None, ax_kws=None, parse_complex='abs', weighted=True):
        """Plot the fit residuals using matplotlib, if available.

        If `yerr` is supplied or if the model included weights, errorbars
        will also be plotted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : str, optional
            Matplotlib format string for data points.
        yerr : numpy.ndarray, optional
            Array of uncertainties for data array.
        data_kws : dict, optional
            Keyword arguments passed on to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed on to the plot function for fitted curve.
        ax_kws : dict, optional
            Keyword arguments for a new axis, if there is one being created.
        parse_complex : str, optional
            How to reduce complex data for plotting.
            Options are one of `['real', 'imag', 'abs', 'angle']`, which
            correspond to the numpy functions of the same name (default is 'abs').
        weighted : Bool
            If True: plot residuals weighted with the given weights. If False or
            no weights are given, plot residuals.

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If `yerr` is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If `yerr` is
        not specified and the fit includes weights, `yerr` set to 1/self.weights

        If model returns complex data, `yerr` is treated the same way that
        weights are in this case.

        If `ax` is None then `matplotlib.pyplot.gca(**ax_kws)` is called.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.

        """
        from matplotlib import pyplot as plt
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if ax_kws is None:
            ax_kws = {}

        # The function reduce_complex will convert complex vectors into real vectors
        reduce_complex = get_reducer(parse_complex)

        if not isinstance(ax, plt.Axes):
            ax = plt.gca(**ax_kws)

        residuals = reduce_complex(self.eval()) - reduce_complex(self.data)
        if weighted and self.weights is not None:
            residuals *= abs(self.weights)

        residuals.name = 'residuals'

        residuals.plot(ax=ax, **fit_kws)

        ax.set_title(self.model.name)
        # ax.legend(loc='best')
        return ax

    def plotly_residuals(self, fig: go.Figure, row: int = 1, col: int = 1,
                         parse_complex='abs', weighted=True, heatmap_args=None):
        if heatmap_args is None:
            heatmap_args = {}
        reduce_complex = get_reducer(parse_complex)
        residuals = reduce_complex(self.eval()) - reduce_complex(self.data)
        if weighted and self.weights is not None:
            residuals *= abs(self.weights)

        residuals.name = 'residuals'
        residuals.plotly_image.add_trace(row=row, col=col, **heatmap_args)

    @_ensureMatplotlib
    def plot(self, datafmt='o', fitfmt='-', initfmt='--', xlabel=None,
             ylabel=None, yerr=None, numpoints=None, fig=None, data_kws=None,
             fit_kws=None, init_kws=None, ax_res_kws=None, ax_fit_kws=None,
             fig_kws=None, show_init=False, parse_complex='abs', fit_color='black',
             init_color='tab:grey', weighted=True):
        """Plot the fit results and residuals using matplotlib, if available.

        The method will produce a matplotlib figure with both results of the
        fit and the residuals plotted. If the fit model included weights,
        errorbars will also be plotted. To show the initial conditions for the
        fit, pass the argument `show_init=True`.

        Parameters
        ----------
        datafmt : str, optional
            Matplotlib format string for data points.
        fitfmt : str, optional
            Matplotlib format string for fitted curve.
        initfmt : str, optional
            Matplotlib format string for initial conditions for the fit.
        xlabel : str, optional
            Matplotlib format string for labeling the x-axis.
        ylabel : str, optional
            Matplotlib format string for labeling the y-axis.
        yerr : numpy.ndarray, optional
            Array of uncertainties for data array.
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. The default is None, which means use the
            current pyplot figure or create one if there is none.
        data_kws : dict, optional
            Keyword arguments passed on to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed on to the plot function for fitted curve.
        init_kws : dict, optional
            Keyword arguments passed on to the plot function for the initial
            conditions of the fit.
        ax_res_kws : dict, optional
            Keyword arguments for the axes for the residuals plot.
        ax_fit_kws : dict, optional
            Keyword arguments for the axes for the fit plot.
        fig_kws : dict, optional
            Keyword arguments for a new figure, if there is one being created.
        show_init : bool, optional
            Whether to show the initial conditions for the fit (default is False).
        parse_complex : str, optional
            How to reduce complex data for plotting.
            Options are one of `['real', 'imag', 'abs', 'angle']`, which
            correspond to the numpy functions of the same name (default is 'abs').
        fit_color : str, optional
            color of the fit contour line, default is black.
        init_color : str, optional
            color of the init contour lines, default is tab:grey.
        weighted : Bool
            If True: plot residuals weighted with the given weights. If False or
            no weights are given, plot residuals.


        Returns
        -------
        A tuple with matplotlib's Figure and GridSpec objects.

        Notes
        -----
        The method combines ModelResult.plot_fit and ModelResult.plot_residuals.

        If `yerr` is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If `yerr` is
        not specified and the fit includes weights, `yerr` set to 1/self.weights

        If model returns complex data, `yerr` is treated the same way that
        weights are in this case.

        If `fig` is None then `matplotlib.pyplot.figure(**fig_kws)` is called,
        otherwise `fig_kws` is ignored.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.

        """
        from matplotlib import pyplot as plt
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_res_kws is None:
            ax_res_kws = {}
        if ax_fit_kws is None:
            ax_fit_kws = {}

        # make a square figure with side equal to the default figure's x-size
        figxsize = plt.rcParams['figure.figsize'][0]
        fig_kws_ = dict(figsize=(figxsize, figxsize))
        if fig_kws is not None:
            fig_kws_.update(fig_kws)

        if not isinstance(fig, plt.Figure):
            fig = plt.figure(**fig_kws_)

        gs = plt.GridSpec(nrows=2, ncols=1, height_ratios=[1, 4])
        ax_res = fig.add_subplot(gs[0], **ax_res_kws)
        ax_fit = fig.add_subplot(gs[1], sharex=ax_res, **ax_fit_kws)

        self.plot_fit(ax=ax_fit, datafmt=datafmt, fitfmt=fitfmt, initfmt=initfmt,
                      fit_color=fit_color, init_color=init_color,
                      xlabel=xlabel, ylabel=ylabel, yerr=yerr, numpoints=numpoints,
                      data_kws=data_kws, fit_kws=fit_kws, init_kws=init_kws, ax_kws=ax_fit_kws,
                      show_init=show_init, parse_complex=parse_complex)
        self.plot_residuals(ax=ax_res, datafmt=datafmt, yerr=yerr, data_kws=data_kws,
                            fit_kws=fit_kws, ax_kws=ax_res_kws, parse_complex=parse_complex, weighted=weighted)
        plt.setp(ax_res.get_xticklabels(), visible=False)
        ax_fit.set_title('')
        return fig, gs
