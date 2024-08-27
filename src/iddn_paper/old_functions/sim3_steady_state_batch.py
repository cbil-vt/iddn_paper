# Improved steady state code for generating samples in batch

import numpy as np
from tqdm import tqdm
import numba
from math import sqrt


# @numba.njit
def calculate_promoter_activity(
    state_prev, parents, parents_roles, n_parent, hill_thr, hill_coef
):
    # the expression probability
    # parent: n_sample x n_node
    f = 1

    if n_parent == 1:
        p = (state_prev[:, parents[0]] / hill_thr) ** hill_coef
        if parents_roles[0] == 0:
            f = p / (1 + p)
        else:
            f = 1 / (1 + p)
    elif n_parent == 2:
        p0 = (state_prev[:, parents[0]] / hill_thr) ** hill_coef
        p1 = (state_prev[:, parents[1]] / hill_thr) ** hill_coef
        pa0_role = parents_roles[0]
        pa1_role = parents_roles[1]
        # only model the AND logic for two regulators
        if pa0_role == 0 and pa1_role == 0:
            f = p0 * p1 / (1 + p0 + p1 + p0 * p1)
        if pa0_role == 0 and pa1_role == 1:
            f = p0 / (1 + p0 + p1 + p0 * p1)
        if pa0_role == 1 and pa1_role == 0:
            f = p1 / (1 + p0 + p1 + p0 * p1)
        if pa0_role == 1 and pa1_role == 1:
            f = 1 / (1 + p0 + p1 + p0 * p1)
    else:
        # only simulate 0 to 2 inputs
        raise ValueError

    return f


# @numba.njit
def calculate_translation_activity(
    state_prev, parents, parents_roles, n_parent, hill_thr, hill_coef
):
    # group parents to mRNA and miRNA
    # parent: n_sample x n_node
    idx_mrna = -1
    idx_mirna = []
    for i in range(n_parent):
        if parents_roles[i] == 2:
            idx_mrna = parents[i]
        if parents_roles[i] == 3:
            idx_mirna.append(parents[i])

    # regulation strength of miRNA
    if len(idx_mirna) == 0:
        f = 0
    elif len(idx_mirna) == 1:
        p = (state_prev[:, idx_mirna[0]] / hill_thr) ** hill_coef
        f = p / (1 + p)
    elif len(idx_mirna) == 2:
        # use OR logic for miRNA
        p0 = (state_prev[:, idx_mirna[0]] / hill_thr) ** hill_coef
        p1 = (state_prev[:, idx_mirna[1]] / hill_thr) ** hill_coef
        f = (p0 + p1 + p0 * p1) / (1 + p0 + p1 + p0 * p1)
    else:
        # TODO: support 3 or 4 miRNAs
        raise ValueError

    return f, idx_mrna


def steady_state_one_step(
    state_history,
    noise_history,
    node_ready_time,
    cur_step,
    idx_layer,
    idx_parent,
    idx_parent_roles,
    n_parent,
    sigma_vec=(),
    sigma_tgt=2.0,
    hill_thr=10,
    hill_coef=1,  # 4
    transcription_rate=20,
    translation_rate=1,
    mrna_degradation_rate=1,
    protein_degradation_rate=1,
):
    # SynTReN like steady state simulation on DAG
    # state_history: n_sample x n_step x n_node
    # The output of each node is scale to mean of hill_thr and std of sigma_tgt

    n_node = len(idx_layer)
    prev_step = cur_step - 1
    state_prev = state_history[:, prev_step]  # n_sample x n_node
    for idx in range(n_node):
        species = idx_layer[idx]
        parents = idx_parent[idx]
        parents_roles = idx_parent_roles[idx]
        n_parent_idx = n_parent[idx]
        # print(idx, parents, parents_roles)

        # if not regulated, consider as external input
        if n_parent_idx == 0:
            state_history[:, cur_step][:, idx] = state_prev[:, idx]
            continue

        # ignore this node if it is already processed
        if node_ready_time[idx] <= prev_step:
            state_history[:, cur_step][:, idx] = state_prev[:, idx]
            continue

        # for current node, check if all it parents are ready
        node_ready = False
        if np.max(node_ready_time[parents[:n_parent_idx]]) <= prev_step:
            node_ready = True
            node_ready_time[idx] = cur_step

        if species == 0 or species == 2:
            # transcription, TF -> mRNA or miRNA
            f = calculate_promoter_activity(
                state_prev,
                parents,
                parents_roles,
                n_parent_idx,
                hill_thr,
                hill_coef,
            )
            x = f * transcription_rate / mrna_degradation_rate
            if node_ready:
                noise = np.random.randn(*x.shape) * sigma_vec[idx]
                noise_history[:, cur_step][:, idx] = noise
                xn = x + noise
                xn[xn < 0] = 0
                xn1 = (xn - np.mean(xn)) / np.std(xn) * sigma_tgt + hill_thr
                state_history[:, cur_step][:, idx] = xn1
            else:
                state_history[:, cur_step][:, idx] = x
        elif species == 1:
            # translation, mRNA + miRNA -> protein
            # miRNA reduce the translation rate
            mirna_activity, idx_mrna = calculate_translation_activity(
                state_prev,
                parents,
                parents_roles,
                n_parent_idx,
                hill_thr,
                hill_coef,
            )
            p = (
                (translation_rate - mirna_activity)
                * state_prev[:, idx_mrna]
                / protein_degradation_rate
            )
            if node_ready:
                noise = np.random.randn(*p.shape) * sigma_vec[idx]
                noise_history[:, cur_step][:, idx] = noise
                pn = p + noise
                pn[pn < 0] = 0
                pn1 = (pn - np.mean(pn)) / np.std(pn) * sigma_tgt + hill_thr
                state_history[:, cur_step][:, idx] = pn1
            else:
                state_history[:, cur_step][:, idx] = p
        else:
            raise ValueError


def run_sim(
    idx_layer,
    idx_parent,
    idx_parent_roles,
    sigma_in=2.0,
    sigma_mid=(2.0,),
    n_sample=1,
    n_max_steps=20,
    method="steady",
    dt=0.02,
    hill_coef=1,
):
    # integrator for each molecule
    n_node = len(idx_layer)

    # make the parent and role dictionary arrays for faster calculation
    idx_layer_arr = np.zeros(len(idx_layer))
    for i, x in idx_layer.items():
        idx_layer_arr[i] = x

    n_parent = np.zeros(n_node, dtype=int)
    idx_parent_arr = np.zeros((n_node, 3), dtype=int) - 1
    for i, x in idx_parent.items():
        idx_parent_arr[i, : len(x)] = x
        n_parent[i] = len(x)
    idx_parent_roles_arr = np.zeros((n_node, 3), dtype=int) - 1
    for i, x in idx_parent_roles.items():
        idx_parent_roles_arr[i, : len(x)] = x

    # record the step when all parents are ready
    # if all parents are ready, add noise
    BIG_VALUE = 10000000
    node_ready_time = np.zeros(n_node) + BIG_VALUE
    node_ready_time[n_parent == 0] = 0

    # initial state
    noise_history = np.zeros((n_sample, n_max_steps, n_node))
    state_history = np.zeros((n_sample, n_max_steps, n_node))

    xx = np.random.randn(n_sample, n_node) * sigma_in + 10
    xx[xx < 0] = 0
    state_history[:, 0] = xx

    cur_step = 0
    for cur_step in range(1, n_max_steps):
        steady_state_one_step(
            state_history,
            noise_history,
            node_ready_time,
            cur_step,
            idx_layer_arr,
            idx_parent_arr,
            idx_parent_roles_arr,
            n_parent,
            hill_coef=hill_coef,
            sigma_vec=sigma_mid,
            sigma_tgt=sigma_in,
        )
        if np.max(node_ready_time) < BIG_VALUE:
            break

    dat_network = state_history[:, cur_step]

    return dat_network, state_history, noise_history
