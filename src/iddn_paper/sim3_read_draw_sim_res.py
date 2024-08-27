"""
Read the simulation results and make figures
"""

import numpy as np
import matplotlib.pyplot as plt
from iddn_paper import sim3_h5op, tool_draw, tool_sys, tool_eval


def read_one_res(
    exp_name,
    n_sample,
    msk_level=2,
    msk_level_in=0,
    mrna_only=False,
    method=None,
    subset=None,
    top_folder=tool_sys.get_work_folder() + "sim_iddn_data",
    top_folder_R=tool_sys.get_work_folder() + "sim_iddn_data",
):
    # Ground truth iddn_data
    dat_file = f"{top_folder}/sim_input/{exp_name}.hdf5"

    (
        _,
        _,
        con_mat1,
        con_mat2,
        comm_gt,
        diff_gt,
        dep_mat_null,
        dep_mat_prior,
        dep_mat_prior_loose,
        layer_count,
    ) = sim3_h5op.read_sim_data(dat_file)

    # Evaluate part of the network (optinal)
    n_node = con_mat1.shape[-1]
    n_gene = layer_count[0]
    # n_gene = layer_count[0] + layer_count[1]
    if mrna_only:
        msk_genes = np.zeros((n_node, n_node))
        msk_genes[:n_gene, :n_gene] = 1
    else:
        msk_genes = np.ones((n_node, n_node))

    if dep_mat_prior_loose.shape[1] > 2:  # fine grained masks
        msk = dep_mat_prior_loose[:, msk_level]
    else:
        if msk_level == 2:
            msk = dep_mat_prior
        elif msk_level == 1:
            msk = dep_mat_prior_loose
        else:
            msk = dep_mat_null

    for i in range(len(msk)):
        msk0 = msk[i] * msk_genes
        msk[i] = msk0

    # Simulation results
    sigma_add = 0.0
    if method == "ddn":
        f_ddn = f"{top_folder}/sim_output/{exp_name}_ddn_sample_{n_sample}_sigma_{sigma_add}.hdf5"
        dep_ddn = tool_eval.read_res_h5(f_ddn, tt=False)
        res_comm, res_diff, res_g1, res_g2, dep_est, _ = tool_eval.gather_res(
            dep_ddn, comm_gt, diff_gt, con_mat1, con_mat2, msk_in=msk
        )
    elif method == "iddn":
        if subset is None:
            f_iddn = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample}_sigma_{sigma_add}_msk_{msk_level_in}.hdf5"
            # f_iddn = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample}_sigma_{sigma_add}_prior_precise.hdf5"
        else:
            f_iddn = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample}_sigma_0.0_msk_0_{subset}.hdf5"
            # f_iddn = f"{top_folder}/sim_output/{exp_name}_iddn_sample_{n_sample}_sigma_0.0_prior_null_{subset}.hdf5"
        dep_iddn = tool_eval.read_res_h5(f_iddn, tt=False)
        res_comm, res_diff, res_g1, res_g2, dep_est, _ = tool_eval.gather_res(
            dep_iddn, comm_gt, diff_gt, con_mat1, con_mat2, msk_in=msk
        )
    elif method == "coe":
        f_coe = f"{top_folder}/sim_output/{exp_name}_coexpression_sample_{n_sample}_sigma_{sigma_add}.hdf5"
        dep_coe = tool_eval.read_res_h5(f_coe, tt=False)
        res_comm, res_diff, res_g1, res_g2, dep_est, res_g1_lst = tool_eval.gather_res(
            dep_coe, comm_gt, diff_gt, con_mat1, con_mat2, msk_in=msk
        )
    elif method == "jgl":
        f_jgl = f"{top_folder_R}/sim_output/{exp_name}_jgl_sample_{n_sample}_sigma_{sigma_add}.hdf5"
        dep_jgl = tool_eval.read_res_h5(f_jgl, tt=True)
        res_comm, res_diff, res_g1, res_g2, dep_est, res_g1_lst = tool_eval.gather_res(
            dep_jgl, comm_gt, diff_gt, con_mat1, con_mat2, msk_in=msk
        )
    elif method == "idingo":
        f_idingo = f"{top_folder_R}/sim_output/{exp_name}_idingo_sample_{n_sample}_sigma_{sigma_add}_3as2_layer.hdf5"
        diff_idingo = tool_eval.read_res_h5(f_idingo, tt=True, key="diffscore")
        g1_idingo = tool_eval.read_res_h5(f_idingo, tt=True, key="R1")
        g2_idingo = tool_eval.read_res_h5(f_idingo, tt=True, key="R2")
        res_comm, res_diff = tool_eval.gather_res_dingo(
            g1_idingo,
            g2_idingo,
            diff_idingo,
            comm_gt,
            diff_gt,
            msk_in=msk,
            n_thr=40,  # 40 to match the lambda1 numbers of iDDN and JGL
        )
        res_g1 = []
        res_g2 = []
    else:
        raise ("Method not simulated")

    return res_comm, res_diff, res_g1, res_g2


def draw_F1_pROC_curves_for_methods(
    comm_lst,
    diff_lst,
    figsize=(10, 8),
    xrange=None,
    xlim=None,
    xlim_1d=None,
    ylim_1d=None,
):
    f1_cdn_lst = []
    f1_ddn_lst = []
    f1_avg_lst = []
    tpfp_cdn_lst = []
    # tpfp_ddn_lst = []
    for n in range(len(comm_lst)):
        res_comm = comm_lst[n]
        res_diff = diff_lst[n]
        f1_rho12_comm = tool_eval.res_to_f1(res_comm)
        f1_rho12_diff = tool_eval.res_to_f1(res_diff)
        f1_rho12_avg = (f1_rho12_diff + f1_rho12_comm) / 2

        f1_rho1_comm = np.max(f1_rho12_comm, axis=1)
        f1_rho1_diff = np.max(f1_rho12_diff, axis=1)
        f1_rho1_avg = np.max(f1_rho12_avg, axis=1)

        f1_cdn_lst.append(f1_rho1_comm)
        f1_ddn_lst.append(f1_rho1_diff)
        f1_avg_lst.append(f1_rho1_avg)

        f1xx_rho12_comm = tool_eval.res_to_f1xx(res_comm)
        f1xx_rho2_comm = np.max(f1xx_rho12_comm, axis=0)
        idx_rho2_comm = np.argmax(f1xx_rho2_comm)
        tpfp_cdn_lst.append(res_comm[:, [idx_rho2_comm], :])

        # f1xx_rho12_diff = tool_eval.res_to_f1xx(res_diff)
        # f1xx_rho2_diff = np.max(f1xx_rho12_diff, axis=0)
        # idx_rho2_diff = np.argmax(f1xx_rho2_diff)
        # tpfp_ddn_lst.append(res_diff[:, [idx_rho2_diff], :])

    # CDN TP-FP
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, constrained_layout=True)
    # fig.tight_layout()

    tool_draw.draw_lines_rho1_rho2_v1(
        res_lst=tpfp_cdn_lst,
        title="pROC of CDN",
        idx1=3,  # FPR
        idx2=2,  # TPR (recall)
        line_width=1.0,
        marker_size=0,
        xlim=xlim,
        ax=ax[1, 1],
        xlabel="FPR",
        ylabel="TPR",
    )

    tool_draw.draw_1d_lines(
        f1_cdn_lst,
        xrange,
        title="F1 of CDN",
        ax=ax[1, 0],
        xlabel="$\lambda_1$",
        ylabel="$F_1$",
        xlim=xlim_1d,
        ylim=ylim_1d,
    )

    tool_draw.draw_1d_lines(
        f1_ddn_lst,
        xrange,
        title="F1 of DDN",
        ax=ax[0, 0],
        xlabel="$\lambda_1$",
        ylabel="$F_1$",
        xlim=xlim_1d,
        ylim=ylim_1d,
    )

    tool_draw.draw_1d_lines(
        f1_avg_lst,
        xrange,
        title="Average F1",
        ax=ax[0, 1],
        xlabel="$\lambda_1$",
        ylabel="$F_1$",
        xlim=xlim_1d,
        ylim=ylim_1d,
    )

    return fig
