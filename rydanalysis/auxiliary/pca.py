import xarray as xr
from sklearn import decomposition
import numpy as np


@xr.register_dataarray_accessor("pca")
class PCAAccessor:
    def __init__(self, reference_images):
        self.reference_images = reference_images

    def __call__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        pca = PCAXarray(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        return pca.fit(self.reference_images)

    @classmethod
    def from_array(cls, reference_images):
        reference_images = xr.DataArray(reference_images, dims=["shot", "x", "y"])
        return cls(reference_images)

    @staticmethod
    def stack(images):
        if not isinstance(images, xr.DataArray):
            images = xr.DataArray(images, dims=["shot", "x", "y"])
        return images.stack({"image_coords": ["x", "y"]})

    def find_references(
        self,
        images,
        mask=None,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        pca_ = self(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

        return pca_.find_references(images, mask)


class PCAXarray(decomposition.PCA):
    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        super().__init__(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        self._dims = ["shot", "image_coords"]
        self._coords = None

    @staticmethod
    def stack(images):
        return images.stack({"image_coords": images.dims[1:]}).dropna("image_coords")

    def fit(self, images, y=None):
        # stack images and cache dims and coords
        stacked_images = self.stack(images)
        self._dims = stacked_images.dims
        self._coords = stacked_images.coords

        # use sklearn's fitting routine
        super().fit(stacked_images)

        # Transform components_ to xr.Dataarray
        components_ = xr.DataArray(
            self.components_,
            dims=["component", "image_coords"],
            coords={"image_coords": self._coords["image_coords"]},
        )
        self.components_ = components_.unstack("image_coords")

        # Transform mean_ to xr.Dataarray
        mean_ = xr.DataArray(
            self.mean_,
            dims=["image_coords"],
            coords={"image_coords": self._coords["image_coords"]},
        )
        self.mean_ = mean_.unstack("image_coords")

        # Transform explained_variance_ to xr.DataArray
        self.explained_variance_ = xr.DataArray(
            self.explained_variance_, dims=["component"]
        )

        return self

    def transform(self, images):
        """Apply dimensionality reduction to X.
        images is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        images : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        images_transformed : array-like, shape (n_samples, n_components)
        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> from rydanalysis import PCAXarray
        >>> images = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> images = xr.DataArray(images, dims=['shot', 'x', 'y'])
        >>> x_pca = PCAXarray(n_components=2, batch_size=3)
        >>> x_pca.fit(images)
        >>> x_pca.transform(images) # doctest: +SKIP
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)

        # images = check_array(images)
        if self.mean_ is not None:
            images = images - self.mean_

        stacked_images = self.stack(images)
        stacked_components = self.stack(self.components_)
        mask = np.logical_not(stacked_images.isnull())
        mask = mask.all([dim for dim in mask.dims if dim is not "image_coords"])

        masked_images = stacked_images.dropna("image_coords")
        masked_components = stacked_components.where(mask, drop=True)
        images_transformed = np.linalg.lstsq(
            masked_components.T, masked_images.T, rcond=None
        )[0]

        not_image_dims = [dim for dim in stacked_images.dims if dim != "image_coords"]
        images_transformed = xr.DataArray(
            images_transformed,
            dims=["component"] + not_image_dims,
            coords={dim: images.coords.get(dim) for dim in not_image_dims},
        )

        if self.whiten:
            images_transformed /= np.sqrt(self.explained_variance_)
        return images_transformed

    def inverse_transform(self, images_transformed):
        """Transform data back to its original space.
        In other words, return an input images whose transform would be images_transformed.
        Parameters
        ----------
        images_transformed : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        Returns
        -------
        images array-like, shape (n_samples, n_features)
        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        if self.whiten:
            return (
                images_transformed.dot(
                    np.sqrt(self.explained_variance_) * self.components_, "component"
                )
                + self.mean_
            )
        else:
            return images_transformed.dot(self.components_, "component") + self.mean_

    # def inverse_transform(self, components):
    #     stacked_images = super().inverse_transform(components)
    #     images = xr.DataArray(stacked_images, dims=self._dims, coords=self._coords)
    #     return images.unstack(self._dims[1])

    def find_references(self, images, mask=None):
        if mask is not None:
            images = images.where(mask)
        images_transformed = self.transform(images)
        return self.inverse_transform(images_transformed)
