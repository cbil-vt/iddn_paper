import numpy as np
import h5py
from ddn3 import tools, performance
import numba


def scan_error_measure_per_condition(
    t1_lst, t2_lst, g1_gt, g2_gt, msk_out=None, msk_in=None
):
    res_g1 = np.zeros((len(t1_lst), 5))
    res_g2 = np.zeros((len(t1_lst), 5))
    res_g1[:] = np.nan
    res_g2[:] = np.nan

    if msk_out is None:
        msk_out = np.zeros_like(g1_gt)
    if msk_in is None:
        msk_in = np.ones_like(g1_gt)

    # msk_in = msk_in + msk_in.T
    msk_in = 1 * (msk_in > 0)
    # msk_out = msk_out + msk_out.T
    msk_out = 1 * (msk_out > 0)

    g1_gt[msk_out > 0] = 0
    g2_gt[msk_out > 0] = 0
    g1_gt = g1_gt * msk_in
    g2_gt = g2_gt * msk_in

    for i in range(len(t1_lst)):
        t1 = t1_lst[i]
        t2 = t2_lst[i]
        if np.isnan(t1[0, 0]):
            print("NaN found, ignore")
            continue
        g1_est = tools.get_net_topo_from_mat(t1)
        g2_est = tools.get_net_topo_from_mat(t2)
        g1_est = g1_est * msk_in
        g2_est = g2_est * msk_in

        g1_est[msk_out > 0] = 0
        g2_est[msk_out > 0] = 0
        res_g1[i] = performance.get_error_measure_two_theta(g1_est, g1_gt)
        res_g2[i] = performance.get_error_measure_two_theta(g2_est, g2_gt)
    return res_g1, res_g2


def scan_error_measure_comm_diff(t1_lst, t2_lst, comm_gt, diff_gt, msk_in=None):
    # The mask may look like np.ix_(np.arange(2,5), np.arange(3,6))
    res_comm = np.zeros((len(t1_lst), 5))
    res_diff = np.zeros((len(t1_lst), 5))
    res_comm[:] = np.nan
    res_diff[:] = np.nan

    if msk_in is None:
        msk_in = np.ones_like(comm_gt)
    # msk_in = msk_in + msk_in.T  # !!!
    msk_in = 1 * (msk_in > 0)
    comm_gt = comm_gt * msk_in
    diff_gt = diff_gt * msk_in

    for i in range(len(t1_lst)):
        t1 = t1_lst[i]
        t2 = t2_lst[i]
        if np.isnan(t1[0, 0]):
            print("NaN found, ignore")
            continue
        comm_est, diff_est = tools.get_common_diff_net_topo([t1, t2])
        comm_est = comm_est * msk_in
        diff_est = diff_est * msk_in
        res_comm[i] = performance.get_error_measure_two_theta(comm_est, comm_gt)
        res_diff[i] = performance.get_error_measure_two_theta(diff_est, diff_gt)
    return res_comm, res_diff


def read_res_h5(h5_file, tt=True, key="dep_est"):
    f = h5py.File(h5_file, "r")
    print(list(f.keys()))
    dep_est = np.array(f[key])  # diffscore for DINGO
    f.close()
    if tt:  # True for HDF5 from R
        dep_est = np.transpose(dep_est, np.arange(len(dep_est.shape) - 1, -1, -1))
    print(dep_est.shape)
    return dep_est


def gather_res(dep_est, comm_gt, diff_gt, con_mat1, con_mat2, msk_in):
    n_rep, n_rho1, n_rho2, _, _, _ = dep_est.shape

    res_comm_lst = np.zeros((n_rep, n_rho1, n_rho2, 5))
    res_diff_lst = np.zeros((n_rep, n_rho1, n_rho2, 5))
    res_g1_lst = np.zeros((n_rep, n_rho1, n_rho2, 5))
    res_g2_lst = np.zeros((n_rep, n_rho1, n_rho2, 5))

    for n in range(n_rep):
        # print(n)
        for j in range(n_rho2):
            t1_lst = dep_est[n, :, j, 0]
            t2_lst = dep_est[n, :, j, 1]

            msk_now = np.copy(msk_in[n])
            msk_now[np.arange(len(msk_now)), np.arange(len(msk_now))] = 0
            # msk_now = msk_now + msk_now.T  # !!!

            res_comm, res_diff = scan_error_measure_comm_diff(
                t1_lst, t2_lst, comm_gt[n], diff_gt[n], msk_in=msk_now
            )
            res_g1, res_g2 = scan_error_measure_per_condition(
                t1_lst, t2_lst, con_mat1[n], con_mat2[n], msk_in=msk_now
            )
            res_comm_lst[n, :, j] = res_comm
            res_diff_lst[n, :, j] = res_diff
            res_g1_lst[n, :, j] = res_g1
            res_g2_lst[n, :, j] = res_g2

    res_comm_mean = np.nanmean(res_comm_lst, axis=0)
    res_diff_mean = np.nanmean(res_diff_lst, axis=0)
    res_g1_mean = np.nanmean(res_g1_lst, axis=0)
    res_g2_mean = np.nanmean(res_g2_lst, axis=0)

    return res_comm_mean, res_diff_mean, res_g1_mean, res_g2_mean, dep_est, res_g1_lst


def gather_res_dingo(
    g1_est_in,
    g2_est_in,
    diff_est_in,
    comm_gt,
    diff_gt,
    msk_in=None,
    n_thr=40,
):
    n_rep = diff_est_in.shape[0]
    n_feature = diff_est_in.shape[1]

    # Differential network
    x_max = np.max(np.abs(diff_est_in))
    thr_diff_lst = np.linspace(
        0.01, x_max, num=n_thr
    )  # 40 to match the lambda1 numbers of iDDN and JGL
    res_diff_lst = np.zeros((n_rep, len(thr_diff_lst), 1, 5))
    for n in range(n_rep):
        for i in range(len(thr_diff_lst)):
            diff_est_thr = 1 * (np.abs(diff_est_in[n]) > thr_diff_lst[i])
            diff_est_thr[np.arange(n_feature), np.arange(n_feature)] = 0
            res_diff = scan_error_measure_diff_dingo(
                diff_est_thr, diff_gt[n], msk_in[n]
            )
            res_diff_lst[n, i, 0] = res_diff
    res_diff_mean = np.mean(res_diff_lst, axis=0)

    # Common network
    thr_corr_lst = np.linspace(0.0, 1.0, num=n_thr)
    res_comm_lst = np.zeros((n_rep, len(thr_corr_lst), 1, 5))
    for n in range(n_rep):
        for i in range(len(thr_corr_lst)):
            g1_est_thr = 1 * (np.abs(g1_est_in[n]) > thr_corr_lst[i])
            g2_est_thr = 1 * (np.abs(g2_est_in[n]) > thr_corr_lst[i])
            comm_est_thr = 1 * ((g1_est_thr + g2_est_thr) == 2)
            comm_est_thr[np.arange(n_feature), np.arange(n_feature)] = 0
            res_comm = scan_error_measure_diff_dingo(
                comm_est_thr, comm_gt[n], msk_in[n]
            )
            res_comm_lst[n, i, 0] = res_comm
    res_comm_mean = np.mean(res_comm_lst, axis=0)

    return res_comm_mean, res_diff_mean


def scan_error_measure_diff_dingo(diff_est, diff_gt, msk_in=None):
    if msk_in is None:
        msk_in = np.ones_like(diff_gt)
    msk_in = msk_in + msk_in.T
    msk_in = 1 * (msk_in > 0)
    diff_est = diff_est * msk_in
    diff_gt = diff_gt * msk_in
    res_diff = performance.get_error_measure_two_theta(diff_est, diff_gt)
    return res_diff


def node_to_idx(mol_names):
    mol_lst = [x.decode("utf-8") for x in mol_names]
    mol_gene_name = [x.split("_")[1] for x in mol_lst]
    mol_gene_idx = np.array([int(x) for x in mol_gene_name])
    n_gene = np.max(mol_gene_idx) + 1
    return mol_gene_idx, n_gene


@numba.njit
def map_dep_to_gene(dep_mat_cur, mol_gene_idx):
    n_mol = len(mol_gene_idx)
    n_gene = np.max(mol_gene_idx) + 1
    dep_mat_gene = np.zeros((n_gene, n_gene))
    for i in range(n_mol):
        for j in range(n_mol):
            i_tgt = mol_gene_idx[i]
            j_tgt = mol_gene_idx[j]
            # Only use absolute values
            dep_mat_gene[i_tgt, j_tgt] = dep_mat_gene[i_tgt, j_tgt] + np.abs(
                dep_mat_cur[i, j]
            )
    res = (dep_mat_gene + dep_mat_gene.T) / 2
    for i in range(n_gene):
        res[i, i] = 0.0
    return res


def map_dep_to_gene_4_dims(dep_in, mol_gene_idx):
    arr_shape = dep_in.shape
    n_gene = np.max(mol_gene_idx) + 1
    new_shape = [*list(arr_shape[:-2]), n_gene, n_gene]
    dep_gene = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                for l in range(new_shape[3]):
                    x_in = dep_in[i, j, k, l]
                    x_out = map_dep_to_gene(x_in, mol_gene_idx)
                    dep_gene[i, j, k, l] = x_out
    return dep_gene


def res_to_f1(res_diff):
    recall = res_diff[:, :, 2]
    prec = res_diff[:, :, 4]
    return performance.get_f1(recall, prec)


def res_to_f1xx(res_diff):
    # Define something similar to F1 score for TPR and FPR
    tpr = res_diff[:, :, 2]
    fpr1 = 1 - res_diff[:, :, 3]
    return performance.get_f1(tpr, fpr1)


def get_f1_rho1(res_comm, res_diff):
    f1_rho12_diff = res_to_f1(res_diff)
    f1_rho12_comm = res_to_f1(res_comm)
    f1_rho12_jgl = (f1_rho12_diff + f1_rho12_comm) / 2
    return np.max(f1_rho12_jgl, axis=1)
