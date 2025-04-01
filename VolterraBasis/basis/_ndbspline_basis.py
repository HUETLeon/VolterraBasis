"""
This is the Ndspline Feature for N-dim
"""
import numpy as np
import scipy.stats
from sklearn.base import TransformerMixin
from ._data_describe import DescribeResult
from ._data_describe import quick_describe, minimal_describe
from ._local_features import _get_bspline_basis
from ._local_features import BSplineFeatures
import scipy.interpolate

def _get_ndbspline_basis(knots, degree=4, periodic=False):
    """Get spline coefficients for each basis spline."""
    nknots = len(knots)
    y_dummy = np.zeros(nknots)

    knots, coeffs, degree = scipy.interpolate.make_splprep(knots, y_dummy, k=degree, per=periodic)
    ncoeffs = len(coeffs)
    bsplines = []
    for ispline in range(nknots):
        coeffs = np.asarray([1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)])
        bsplines.append((knots, coeffs, degree))
    return bsplines


class NdBSplineFeatures(BSplineFeatures):  # TODO replace current implementation by one using Bspline.basis_element
    """
    Bsplines features class
    """

    def __init__(self, n_knots=5, k=3, periodic=False, remove_const=True):
        """
        Parameters
        ----------
        n_knots : int
            Number of knots to use
        k : int
            Degree of the splines
        periodic: bool
            Whatever to use periodic splines or not
        """
        self.periodic = periodic
        self.k = k
        self.n_knots = n_knots  # knots are position along the axis of the knots
        self.const_removed = remove_const
        self.dim_out_basis = 1

    def fit(self, describe_result, knots=None):
        if isinstance(describe_result, np.ndarray):
            describe_result = minimal_describe(describe_result)
        dim = describe_result.mean.shape[0]
        # TODO determine non uniform position of knots given the datas
        if knots is None:
            knots = np.linspace(describe_result.minmax[0], describe_result.minmax[1], self.n_knots)
        self.bsplines_ = _get_ndbspline_basis(knots, self.k, periodic=self.periodic)
        self._nsplines = len(self.bsplines_)
        self.n_output_features_ = len(self.bsplines_) * dim
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def deriv(self, X, deriv_order=1):
        nsamples, dim = X.shape
        with_const = int(self.const_removed)
        features = np.zeros((nsamples, dim * (self._nsplines - with_const)) + (dim,) * deriv_order)
        if self.k < deriv_order:
            return features
        for ispline, spline in enumerate(self.bsplines_[: len(self.bsplines_) - with_const]):
            istart = (ispline) * dim
            for i in range(dim):
                features[(Ellipsis, slice(istart + i, istart + i + 1)) + (i,) * deriv_order] = scipy.interpolate.splev(X[:, slice(i, i + 1)], spline, der=deriv_order)
        return features

    def hessian(self, X):
        return self.deriv(X, deriv_order=2)

    def antiderivative(self, X, order=1):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, self.n_output_features_))
        for ispline, spline in enumerate(self.bsplines_):
            istart = ispline * dim
            iend = (ispline + 1) * dim
            spline_int = scipy.interpolate.splantider(spline, n=order)
            features[:, istart:iend] = scipy.interpolate.splev(X, spline_int)
        return features
