"""
This the main estimator module
"""
import numpy as np

import scipy.interpolate
import scipy.stats

from sklearn.base import TransformerMixin


class LinearFeatures(TransformerMixin):
    """
    Linear function
    """

    def __init__(self, to_center=False):
        """"""
        self.centered = to_center

    def fit(self, X, y=None):
        self.n_output_features_ = X.shape[1]
        if self.centered:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros((self.n_output_features_,))
        return self

    def basis(self, X):
        return X - self.mean_

    def deriv(self, X, deriv_order=1, remove_const=True):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        if deriv_order == 1:
            for i in range(dim):
                grad[:, i, i] = 1.0
        return grad

    def hessian(self, X, remove_const=True):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)


class PolynomialFeatures(TransformerMixin):
    """
    Wrapper for numpy polynomial series removing the constant polynom
    """

    def __init__(self, deg=1, polynom=np.polynomial.Polynomial):
        self.degree = deg + 1
        self.polynom = polynom

    def fit(self, X, y=None):
        self.n_output_features_ = X.shape[1] * self.degree
        return self

    def basis(self, X):
        nsamples, dim = X.shape
        features = np.zeros((nsamples, dim * self.degree))
        for n in range(0, self.degree):
            istart = n * dim
            iend = (n + 1) * dim
            features[:, istart:iend] = self.polynom.basis(n)(X)
        return features

    def deriv(self, X, deriv_order=1, remove_const=False):
        nsamples, dim = X.shape
        with_const = int(remove_const)
        features = np.zeros((nsamples, dim * (self.degree - with_const)) + (dim,) * deriv_order)
        for n in range(self.degree - with_const):
            istart = (n) * dim
            iend = (n + 1) * dim
            # print(self.polynom.basis(n).deriv(deriv_order)(X).shape)
            # print(features.shape)
            for i in range(dim):
                features[(Ellipsis, slice(istart, iend)) + (i,) * deriv_order] = self.polynom.basis(n).deriv(deriv_order)(X[:, slice(i, i + 1)])
        return features

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)


class SplineFctFeatures(TransformerMixin):
    """
    A single basis function that is given from splines fit of data
    """

    def __init__(self, knots, coeffs, k=3, periodic=False):
        self.periodic = periodic
        self.k = k
        self.t = knots  # knots are position along the axis of the knots
        self.c = coeffs

    def fit(self, X, y=None):
        nsamples, dim = X.shape
        self.spl = scipy.interpolate.BSpline(self.t, self.c, self.k)
        self.n_output_features_ = dim
        return self

    def basis(self, X):
        return self.spl(X)

    def deriv(self, X, deriv_order=1, remove_const=False):
        nsamples, dim = X.shape
        grad = np.zeros((nsamples, dim) + (dim,) * deriv_order)
        for i in range(dim):
            grad[(Ellipsis,) + (i,) * deriv_order] = self.spl.derivative(deriv_order)(X[:, slice(i, i + 1)])
        return grad

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)

    def antiderivative(self, X, order=1):
        return self.spl.antiderivative(order)(X)


class FeaturesCombiner(TransformerMixin):
    def __init__(self, *basis):
        self.basis_set = basis

    def fit(self, X, y=None):
        for b in self.basis_set:
            b.fit(X)
        self.n_output_features_ = np.sum([b.n_output_features_ for b in self.basis_set])
        return self

    def basis(self, X):
        features = self.basis_set[0].basis(X)
        for b in self.basis_set[1:]:
            features = np.concatenate((features, b.basis(X)), axis=1)
        return features

    def deriv(self, X, deriv_order=1, remove_const=False):
        grad = self.basis_set[0].deriv(X, deriv_order=deriv_order, remove_const=remove_const)
        for b in self.basis_set[1:]:
            print(grad.shape, b.deriv(X, deriv_order=deriv_order, remove_const=remove_const).shape)
            features = np.concatenate((grad, b.deriv(X, deriv_order=deriv_order, remove_const=remove_const)), axis=1)
        return features

    def hessian(self, X, remove_const=False):
        return self.deriv(X, deriv_order=2, remove_const=remove_const)


# class SplineFctWithLinFeatures(TransformerMixin):
#     """
#     Combine a basis function that is given from splines fit of data with linear function
#     """
#
#     def __init__(self, knots, coeffs, k=3, periodic=False):
#         self.periodic = periodic
#         self.k = k
#         self.t = knots  # knots are position along the axis of the knots
#         self.c = coeffs
#
#     def fit(self, X, y=None):
#         nsamples, dim = X.shape
#         self.spl = scipy.interpolate.BSpline(self.t, self.c, self.k)
#         self.n_output_features_ = 2 * dim
#         return self
#
#     def basis(self, X):
#         return np.concatenate((X, self.spl(X)), axis=1)
#
#     def deriv(self, X, deriv_order=1, remove_const=False):
#         if deriv_order == 1:
#             lin_deriv = np.ones_like(X)
#         else:
#             lin_deriv = np.zeros_like(X)
#         return np.concatenate((lin_deriv, self.spl.derivative(deriv_order)(X)), axis=1)
#
#     def hessian(self, X, remove_const=False):
#         return self.deriv(X, deriv_order=2, remove_const=remove_const)
#
#     def antiderivative(self, X, order=1):
#         return self.spl.antiderivative(order)(X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_range = np.linspace(-10, 10, 30).reshape(-1, 2)
    b2 = LinearFeatures()
    # basis = PolynomialFeatures(deg=3)
    b1 = SplineFctFeatures(knots=np.linspace(-1, 1, 8), coeffs=np.logspace(1, 2, 8), k=2)
    basis = FeaturesCombiner(b1, b2)
    basis.fit(x_range)
    print(x_range.shape)
    print("Basis")
    print(basis.basis(x_range).shape)
    print("Deriv")
    print(basis.deriv(x_range).shape)
    print("Hessian")
    print(basis.hessian(x_range).shape)

    # Plot basis
    x_range = np.linspace(-2, 2, 50).reshape(-1, 1)
    basis = basis.fit(x_range)
    # basis = LinearFeatures().fit(x_range)
    y = basis.basis(x_range)
    plt.grid()
    for n in range(y.shape[1]):
        plt.plot(x_range[:, 0], y[:, n])
    plt.show()
