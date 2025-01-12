#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
=====================================
Kernel Estimation for 2D observable
=====================================

How to run kernel estimation
"""

import numpy as np
import matplotlib.pyplot as plt

import VolterraBasis as vb
import VolterraBasis.basis as bf

trj = np.loadtxt("example_2d.trj")
xva_list = []
print(trj.shape)
# for i in range(1, trj.shape[1]):
#     xf = vb.xframe(trj[:, i], trj[:, 0])
#     xvaf = vb.compute_va(xf)
#     xva_list.append(xvaf)

xf = vb.xframe(trj[:, (1, 3)], trj[:, 0] - trj[0, 0])
xvaf = vb.compute_va(xf)
xva_list.append(xvaf)


estimator = vb.Estimator_gle(xva_list, vb.Pos_gle, bf.TensorialBasis2D(bf.PolynomialFeatures(deg=1)), trunc=10, saveall=False)
print("Dimension of observable", estimator.model.dim_x)
model = estimator.compute_mean_force()
# print(mymem.force_coeff)
print(model.N_basis_elt, model.N_basis_elt_force, model.N_basis_elt_kernel)
# print(mymem.basis.b1.n_output_features_, mymem.basis.b2.n_output_features_)
estimator.compute_corrs()
model = estimator.compute_kernel(method="trapz")
kernel = model.kernel_eval([[1.5, 1.0], [2.0, 1.5], [2.5, 1.0]])
time = kernel["time_kernel"]
print(time.shape, kernel.shape)
# To find a correct parametrization of the space
bins = np.histogram_bin_edges(xvaf["x"], bins=15)
xfa = (bins[1:] + bins[:-1]) / 2.0
x, y = np.meshgrid(xfa, xfa)
X = np.vstack((x.flatten(), y.flatten())).T
force = model.force_eval(X)

# Compute noise
time_noise, noise_reconstructed, _, _, _ = model.compute_noise(xva_list[0], trunc_kernel=200)


fig_kernel, axs = plt.subplots(1, 3)
# Force plot
axs[0].set_title("Force")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$y$")
# axs[0].grid()
axs[0].quiver(x, y, force[:, 0], force[:, 1])
# Kernel plot
axs[1].set_title("Memory kernel")
axs[1].set_xscale("log")
axs[1].set_xlabel("$t$")
axs[1].set_ylabel("$\\Gamma$")
axs[1].grid()
axs[1].plot(time, kernel[:, :, 0, 0], "-x")
axs[1].plot(time, kernel[:, :, 0, 1], "-x")
axs[1].plot(time, kernel[:, :, 1, 0], "-x")
axs[1].plot(time, kernel[:, :, 1, 1], "-x")

# Noise plot
axs[2].set_title("Noise")
axs[2].set_xlabel("$t$")
axs[2].set_ylabel("$\\xi_t$")
axs[2].grid()
axs[2].plot(time_noise, noise_reconstructed, "-")
plt.show()
