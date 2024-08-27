"""
Tools for running simulations for iDDN and DDN3.0 using different hyper-parameters
"""

import numpy as np
from ddn3 import ddn
from iddn import iddn


def scan_ddn(dat1, dat2, lambda1_rg, lambda2=0.1, mthd="resi"):
    t1_lst = []
    t2_lst = []
    for i, lamb in enumerate(lambda1_rg):
        out_ddn = ddn.ddn(
            dat1, dat2, lambda1=lamb, lambda2=lambda2, threshold=1e-5, mthd=mthd
        )
        t1_lst.append(np.copy(out_ddn[0]))
        t2_lst.append(np.copy(out_ddn[1]))
    return t1_lst, t2_lst


def scan_iddn(
    dat1,
    dat2,
    dep_mat,
    rho1_mat,
    rho2_mat,
    lambda1_rg,
    lambda2=0.1,
    mthd="resi",
):
    t1_lst = []
    t2_lst = []
    for i, lambda1 in enumerate(lambda1_rg):
        # print(i, lambda1)
        lambda1_mat = rho1_mat * lambda1
        lambda2_mat = rho2_mat * lambda2
        out_ddn = iddn.iddn(
            dat1,
            dat2,
            lambda1=lambda1_mat,
            lambda2=lambda2_mat,
            dep_mat=dep_mat,
            mthd=mthd,
        )
        t1_lst.append(np.copy(out_ddn[0]))
        t2_lst.append(np.copy(out_ddn[1]))
    return t1_lst, t2_lst


def scan2_ddn(dat1, dat2, rho1_rg, rho2_rg, n_sample_work, sigma_add=0, n=0):
    # Repeat, lambda1, lambda2, conditions, feature, feature
    _, n_feature = dat1.shape
    # idx1 = np.random.choice(n_sample, n_sample_work, replace=False)
    # idx2 = np.random.choice(n_sample, n_sample_work, replace=False)
    # idx1 = np.arange(n_sample_work)
    # idx2 = np.arange(n_sample_work)
    dat1_sel = dat1[:n_sample_work, :]
    dat2_sel = dat2[:n_sample_work, :]
    dat1_sel = dat1_sel + np.random.normal(0, sigma_add, dat1_sel.shape)
    dat2_sel = dat2_sel + np.random.normal(0, sigma_add, dat2_sel.shape)
    res_mat0 = np.zeros((len(rho1_rg), len(rho2_rg), 2, n_feature, n_feature))

    for j, rho2 in enumerate(rho2_rg):
        print(n, rho2)
        t1_lst, t2_lst = scan_ddn(dat1_sel, dat2_sel, rho1_rg, lambda2=rho2)
        res_mat0[:, j, 0] = np.array(t1_lst)
        res_mat0[:, j, 1] = np.array(t2_lst)
    return res_mat0


def scan2_iddn(
    dat1,
    dat2,
    rho1_rg,
    rho2_rg,
    dep_mat=None,
    n_sample_work=100,
    sigma_add=0.0,
    n=0,
    make_prior_symm=True,
    mthd="resi",
):
    # Repeat, lambda1, lambda2, conditions, feature, feature
    _, n_feature = dat1.shape
    # idx1 = np.random.choice(n_sample, n_sample_work, replace=False)
    # idx2 = np.random.choice(n_sample, n_sample_work, replace=False)
    # dat1_sel = dat1[idx1, :]
    # dat2_sel = dat2[idx2, :]
    dat1_sel = dat1[:n_sample_work, :]
    dat2_sel = dat2[:n_sample_work, :]
    dat1_sel = dat1_sel + np.random.normal(0, sigma_add, dat1_sel.shape)
    dat2_sel = dat2_sel + np.random.normal(0, sigma_add, dat2_sel.shape)

    if dep_mat is None:
        dep_mat = np.ones((n_feature, n_feature))
    if make_prior_symm:
        dep_mat = dep_mat + dep_mat.T
    rho1_mat = np.ones((n_feature, n_feature))
    rho2_mat = np.ones((n_feature, n_feature))

    res_mat0 = np.zeros((len(rho1_rg), len(rho2_rg), 2, n_feature, n_feature))
    for j, rho2 in enumerate(rho2_rg):
        print(n, rho2)
        t1_lst, t2_lst = scan_iddn(
            dat1_sel,
            dat2_sel,
            dep_mat,
            rho1_mat,
            rho2_mat,
            rho1_rg,
            lambda2=rho2,
            mthd=mthd,
        )
        res_mat0[:, j, 0] = np.array(t1_lst)
        res_mat0[:, j, 1] = np.array(t2_lst)
    return res_mat0
