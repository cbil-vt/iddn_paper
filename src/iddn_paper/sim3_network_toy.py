import numpy as np
from scipy.stats import powerlaw
from ddn3 import simulation
from ddn3_extra import simulation_r


################
# GGM without explicit translation
################


def toy_two_layer(
    n_tf=50,
    n_mrna=50,
    diag_scale=0.55,
    n_sample_gen=500,
    ratio_diff=0.25,
    tf_mrna_powerlaw=True,
    tf_tf_network=True,
    tf_tf_graph_type="scale-free",
):
    n_node = n_tf + n_mrna
    dep_mat_prior = np.ones((n_node, n_node))
    dep_mat_prior[n_tf:, n_tf:] = 0

    if tf_tf_network:
        omega, _, _ = simulation_r.huge_omega(
            n_node=n_tf,
            ratio_diff=0.0,
            graph_type=tf_tf_graph_type,
            n_group=1,
        )
    else:
        omega = np.eye(n_tf)
        dep_mat_prior[:n_tf:, :n_tf] = 0

    omega_ext = np.eye(n_node)
    omega_ext[:n_tf, :n_tf] = 1 * (np.abs(omega) > 1e-3)
    # tf_deg = np.sum(np.abs(omega)>1e-3, axis=1)
    mrna_lst = np.arange(n_tf, n_node)
    n_conn_tot = 0
    for n in range(n_tf):
        if tf_mrna_powerlaw:
            n_conn = int(powerlaw.rvs(0.3) * 10) + 1
        else:
            n_conn = 2
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        omega_ext[tgt, n] = 1.0
        omega_ext[n, tgt] = 1.0
        n_conn_tot += n_conn
    print("Edges: ", n_conn_tot)

    dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, o1m, om2 = post_process(
        omega_ext, dep_mat_prior, ratio_diff, diag_scale, n_sample_gen
    )
    return dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, dep_mat_prior, o1m, om2


def toy_three_layer(
    n_mrna=50,
    n_tf=50,
    n_mirna=50,
    n_sample_gen=500,
    ratio_diff=0.25,
    mrna_graph_type="scale-free",
    ratio_diag=1.0,
    n_conn_tf_mrna=2,
    n_conn_mirna_mrna=2,
):
    # mRNA (network), TF (protein), miRNA
    n_node = n_mrna + n_tf + n_mirna
    mrna_lst = np.arange(n_mrna)
    tf_lst = np.arange(n_mrna, n_mrna + n_tf)
    mirna_lst = np.arange(n_tf + n_mrna, n_tf + n_mrna + n_mirna)

    dep_mat_prior = np.zeros((n_node, n_node))
    dep_mat_prior[:n_mrna, :n_mrna] = 1
    dep_mat_prior[np.ix_(tf_lst, mrna_lst)] = 1
    dep_mat_prior[np.ix_(mirna_lst, mrna_lst)] = 1
    dep_mat_prior = 1 * ((dep_mat_prior + dep_mat_prior.T) > 0)

    omega_ext = np.zeros((n_node, n_node))

    # mRNA network
    omega, _, _ = simulation_r.huge_omega(
        n_node=n_mrna,
        ratio_diff=0.0,
        graph_type=mrna_graph_type,
        n_group=1,
    )
    omega_ext[:n_mrna, :n_mrna] = 1 * (np.abs(omega) > 1e-3)
    for i in range(n_mrna):
        for j in range(i):
            if np.random.rand() > 0.5:
                xx = omega_ext[i, j]
                omega_ext[i, j] = -xx
                omega_ext[j, i] = -xx

    # TF to mRNA
    for n in tf_lst:
        n_conn = n_conn_tf_mrna
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        if np.random.rand() > 0.5:
            omega_ext[tgt, n] = 1.0
            omega_ext[n, tgt] = 1.0
        else:
            omega_ext[tgt, n] = -1.0
            omega_ext[n, tgt] = -1.0

    # miRNA to mRNA
    for n in mirna_lst:
        n_conn = n_conn_mirna_mrna
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        omega_ext[tgt, n] = -1.0
        omega_ext[n, tgt] = -1.0

    omega_ext[np.arange(n_node), np.arange(n_node)] = 0.0

    dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, om1, om2 = post_process(
        omega_ext,
        dep_mat_prior,
        ratio_diff,
        diag_scale=None,
        n_sample_gen=n_sample_gen,
        make_diff="remove",
        ratio_diag=ratio_diag,
    )
    return dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, dep_mat_prior, om1, om2


def toy_three_layer_v2(
    n_mrna=50,
    n_tf=50,
    n_mirna=50,
    n_sample_gen=500,
    ratio_diff=0.25,
    tf_graph_type="scale-free",
    mrna_graph_type="scale-free",
    ratio_diag=1.0,
    n_conn_tf_mrna=2,
    n_conn_mirna_mrna=2,
):
    # Networks in both TFs and mRNAs
    # miRNA to mRNA
    # More detailed constrain levels

    # mRNA (network), TF (protein), miRNA
    n_node = n_mrna + n_tf + n_mirna
    mrna_lst = np.arange(n_mrna)
    tf_lst = np.arange(n_mrna, n_mrna + n_tf)
    mirna_lst = np.arange(n_tf + n_mrna, n_tf + n_mrna + n_mirna)

    # A generic mask
    dep_mat_prior = np.zeros((n_node, n_node))
    dep_mat_prior[:n_mrna, :n_mrna] = 1
    dep_mat_prior[np.ix_(tf_lst, tf_lst)] = 1  # allow TF network
    dep_mat_prior[np.ix_(tf_lst, mrna_lst)] = 1
    dep_mat_prior[np.ix_(mirna_lst, mrna_lst)] = 1
    dep_mat_prior = 1 * ((dep_mat_prior + dep_mat_prior.T) > 0)

    omega_ext = np.zeros((n_node, n_node))

    # mRNA network
    omega_mrna, _, _ = simulation_r.huge_omega(
        n_node=n_mrna,
        ratio_diff=0.0,
        graph_type=mrna_graph_type,
        n_group=1,
    )
    omega_ext[:n_mrna, :n_mrna] = 1 * (np.abs(omega_mrna) > 1e-3)
    for i in range(n_mrna):
        for j in range(i):
            if np.random.rand() > 0.5:
                xx = omega_ext[i, j]
                omega_ext[i, j] = -xx
                omega_ext[j, i] = -xx

    # TF protein network
    omega_tf, _, _ = simulation_r.huge_omega(
        n_node=n_tf,
        ratio_diff=0.0,
        graph_type=tf_graph_type,
        n_group=1,
    )
    omega_ext[np.ix_(tf_lst, tf_lst)] = 1 * (np.abs(omega_tf) > 1e-3)
    for i in range(n_mrna):
        for j in range(i):
            if np.random.rand() > 0.5:
                xx = omega_ext[i, j]
                omega_ext[i, j] = -xx
                omega_ext[j, i] = -xx

    # TF to mRNA
    for n in tf_lst:
        n_conn = n_conn_tf_mrna
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        if np.random.rand() > 0.5:
            omega_ext[tgt, n] = 1.0
            omega_ext[n, tgt] = 1.0
        else:
            omega_ext[tgt, n] = -1.0
            omega_ext[n, tgt] = -1.0

    # miRNA to mRNA
    for n in mirna_lst:
        n_conn = n_conn_mirna_mrna
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        omega_ext[tgt, n] = -1.0
        omega_ext[n, tgt] = -1.0

    omega_ext[np.arange(n_node), np.arange(n_node)] = 0.0

    # more detailed prior
    msk = np.tril(1 * (omega_ext == 0) * (dep_mat_prior != 0))
    idx = np.where(msk > 0)
    n_points = len(idx[0])
    msk_lst = [np.ones_like(dep_mat_prior), dep_mat_prior]
    ratio_rg = np.arange(0.1, 0.91, 0.1)
    for ratio in ratio_rg:
        msk1 = np.copy(dep_mat_prior)
        pts_sel = int(n_points * ratio)
        idx_sel = np.random.choice(n_points, pts_sel, replace=False)
        idx0 = idx[0][idx_sel]
        idx1 = idx[1][idx_sel]
        msk1[idx0, idx1] = 0
        msk1[idx1, idx0] = 0
        msk_lst.append(msk1)

    dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, om1, om2 = post_process(
        omega_ext,
        dep_mat_prior=None,
        ratio_diff=ratio_diff,
        diag_scale=None,
        n_sample_gen=n_sample_gen,
        make_diff="remove",
        ratio_diag=ratio_diag,
    )
    return dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, msk_lst, om1, om2


def toy_four_layer(
    n_tf=50,
    n_mrna=50,
    n_mirna=50,
    n_lncrna=50,
    diag_scale=0.55,
    n_sample_gen=500,
    ratio_diff=0.25,
    tf_mrna_powerlaw=True,
    mirna_tf_powerlaw=True,
    lncrna_mrna_powerlaw=True,
    tf_tf_network=True,
    tf_tf_graph_type="scale-free",
    make_diff="add",
    ratio_diag=1.0,
    n_conn_tf_mrna=2,
    n_conn_mirna_tf=2,
    n_conn_lncrna_mrna=2,
):
    n_node = n_tf + n_mrna + n_mirna + n_lncrna
    tf_lst = np.arange(n_tf)
    mrna_lst = np.arange(n_tf, n_tf + n_mrna)
    mirna_lst = np.arange(n_tf + n_mrna, n_tf + n_mrna + n_mirna)
    lncrna_lst = np.arange(n_tf + n_mrna + n_mirna, n_node)

    omega_ext = np.zeros((n_node, n_node))
    dep_mat_prior = np.zeros((n_node, n_node))
    dep_mat_prior[np.ix_(mrna_lst, tf_lst)] = 1
    dep_mat_prior[np.ix_(mirna_lst, tf_lst)] = 1
    dep_mat_prior[np.ix_(lncrna_lst, mrna_lst)] = 1
    dep_mat_prior = dep_mat_prior + dep_mat_prior.T

    # TF-TF, optional
    if tf_tf_network:
        omega, _, _ = simulation_r.huge_omega(
            n_node=n_tf,
            ratio_diff=0.0,
            graph_type=tf_tf_graph_type,
            n_group=1,
        )
        dep_mat_prior[:n_tf:, :n_tf] = 1
    else:
        omega = np.eye(n_tf)
        dep_mat_prior[:n_tf:, :n_tf] = 0
    omega_ext[:n_tf, :n_tf] = 1 * (np.abs(omega) > 1e-3)

    # Allow both positive and negative values
    for i in range(n_tf):
        for j in range(i):
            if np.random.rand() > 0.5:
                xx = omega_ext[i, j]
                omega_ext[i, j] = -xx
                omega_ext[j, i] = -xx

    # TF to mRNA
    for n in tf_lst:
        if tf_mrna_powerlaw:
            n_conn = int(powerlaw.rvs(0.3) * 10) + 1
        else:
            n_conn = n_conn_tf_mrna
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        if np.random.rand() > 0.5:
            omega_ext[tgt, n] = 1.0
            omega_ext[n, tgt] = 1.0
        else:
            omega_ext[tgt, n] = -1.0
            omega_ext[n, tgt] = -1.0

    # miRNA to TF
    for n in mirna_lst:
        if mirna_tf_powerlaw:
            n_conn = int(powerlaw.rvs(0.3) * 10) + 1
        else:
            n_conn = n_conn_mirna_tf
        tgt = np.random.choice(tf_lst, n_conn, replace=False)
        omega_ext[tgt, n] = -1.0
        omega_ext[n, tgt] = -1.0

    # lncRNA to mRNA
    for n in lncrna_lst:
        if lncrna_mrna_powerlaw:
            n_conn = int(powerlaw.rvs(0.3) * 10) + 1
        else:
            n_conn = n_conn_lncrna_mrna
        tgt = np.random.choice(mrna_lst, n_conn, replace=False)
        if np.random.rand() > 0.5:
            omega_ext[tgt, n] = 1.0
            omega_ext[n, tgt] = 1.0
        else:
            omega_ext[tgt, n] = -1.0
            omega_ext[n, tgt] = -1.0

    omega_ext[np.arange(n_node), np.arange(n_node)] = 0.0

    dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, om1, om2 = post_process(
        omega_ext,
        dep_mat_prior,
        ratio_diff,
        diag_scale=diag_scale,
        n_sample_gen=n_sample_gen,
        make_diff=make_diff,
        ratio_diag=ratio_diag,
    )
    return dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, dep_mat_prior, om1, om2


def post_process(
    omega_ext,
    dep_mat_prior=None,
    ratio_diff=0.25,
    diag_scale=None,
    n_sample_gen=500,
    make_diff="remove",
    ratio_diag=1.0,
):
    n_node = len(omega_ext)

    if diag_scale is None:
        # Same as HUGE
        eigenvalues, eigenvectors = np.linalg.eig(omega_ext)
        v = 0.9
        u = 0.1
        omega_ext1 = omega_ext * v + (np.abs(np.min(eigenvalues)) + 0.1 + u) * np.eye(
            len(omega_ext)
        )
    else:
        omega_ext1 = np.copy(omega_ext)
        xx = np.maximum(
            np.sum(np.abs(omega_ext1), axis=0)[:]
            + np.sum(np.abs(omega_ext1), axis=1)[:],
            1.0,
        )
        omega_ext1[np.arange(len(omega_ext1)), np.arange(len(omega_ext1))] = (
            xx * diag_scale
        )

    if make_diff == "add":
        dep_in = dep_mat_prior + dep_mat_prior.T
        omega1, omega2 = simulation_r.make_two_from_one(
            omega_ext1, dep_in, ratio_diff=ratio_diff, verbose=True
        )
    else:
        omega1, omega2 = simulation_r.make_two_from_one_by_removing(
            omega_ext1, ratio_diff=ratio_diff, ratio_diag=ratio_diag, verbose=True
        )

    g1_cov, g2_cov, comm_gt, diff_gt = simulation.prep_sim_from_two_omega(
        omega1, omega2
    )
    con_mat1 = 1 * (np.abs(omega1) > 1e-8)
    con_mat2 = 1 * (np.abs(omega2) > 1e-8)
    con_mat1[np.arange(n_node), np.arange(n_node)] = 0
    con_mat2[np.arange(n_node), np.arange(n_node)] = 0

    dat1, dat2 = simulation.gen_sample_two_conditions(
        g1_cov, g2_cov, n_sample_gen, n_sample_gen
    )
    return dat1, dat2, con_mat1, con_mat2, comm_gt, diff_gt, omega1, omega2


################
# Model with explicit translation
################


def toy_example():
    # molecular network for TF protein and mRNA
    # Feedback not used here
    # layer. 0: RNA, 1: protein
    # role of a parent: 0: TF activate, 1: TF repress, 2: dosage in translation, 3: miRNA repress

    idx_layer = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    idx_par = {
        0: [],
        1: [5],
        2: [6],
        3: [6, 7],
        4: [7],
        5: [],
        6: [],
        7: [0, 1],
        8: [2],
        9: [3, 4],
    }
    idx_par_roles = {
        0: [],
        1: [0],
        2: [0],
        3: [0, 1],
        4: [1],
        5: [],
        6: [],
        7: [3, 2],
        8: [2],
        9: [3, 2],
    }
    # _, state_history = sim3_ode.run_sim(
    #     idx_layer, idx_par, idx_par_roles, n_sample=1, n_max_steps=1000
    # )
    return idx_layer, idx_par, idx_par_roles


def _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, par_names):
    mrna_name = gene_name + "_mrna"
    prot_name = gene_name + "_prot"
    mol_layer[mrna_name] = 0
    mol_layer[prot_name] = 1
    mol_par[mrna_name] = par_names
    mol_par[prot_name] = [mrna_name]
    if len(par_names) > 0:
        mol_par_roles[mrna_name] = np.random.randint(
            low=0, high=2, size=len(par_names)
        ).tolist()
    else:
        mol_par_roles[mrna_name] = []
    mol_par_roles[prot_name] = [2]


def add_one_hub_net(
    mol_layer, mol_par, mol_par_roles, hub_to_tf, hub_to_gene, tf_to_gene, net_idx
):
    gene_name = f"hub_{net_idx}_tf_hub"
    _add_one_gene(mol_layer, mol_par, mol_par_roles, gene_name, [])

    for i in range(hub_to_gene):
        gene_name = f"hub_{net_idx}_tf_hub_gene_{i}"
        _add_one_gene(
            mol_layer, mol_par, mol_par_roles, gene_name, [f"hub_{net_idx}_tf_hub_prot"]
        )

    for i in range(hub_to_tf):
        gene_name = f"hub_{net_idx}_tf_{i}"
        _add_one_gene(
            mol_layer, mol_par, mol_par_roles, gene_name, [f"hub_{net_idx}_tf_hub_prot"]
        )

    for i in range(hub_to_tf):
        for j in range(tf_to_gene):
            gene_name = f"hub_{net_idx}_tf_{i}_gene_{j}"
            _add_one_gene(
                mol_layer,
                mol_par,
                mol_par_roles,
                gene_name,
                [f"hub_{net_idx}_tf_{i}_prot"],
            )


################
# Utilities
################


def molnet_to_genenet(mol_par):
    # make gene network from molecular network
    # use the names of molecules to group them
    gene_mol = dict()
    for x in mol_par.keys():
        x_base = x[:-5]
        if not x_base in gene_mol:
            gene_mol[x_base] = [x]
        else:
            gene_mol[x_base].append(x)

    gene_par = dict()
    for x, px in mol_par.items():
        x_base = x[:-5]
        if not x_base in gene_par:
            gene_par[x_base] = []
        for px0 in px:
            px0_base = px0[:-5]
            if px0_base not in gene_par[x_base]:
                if px0_base != x_base:
                    gene_par[x_base].append(px0_base)

    gene2idx = dict()
    idx2gene = dict()
    i = 0
    for x in gene_par:
        gene2idx[x] = i
        idx2gene[i] = x
        i += 1

    n_genes = len(gene_par)
    gene_dep_mat = np.zeros((n_genes, n_genes))
    for x, px in gene_par.items():
        for px0 in px:
            x_idx = gene2idx[x]
            px0_idx = gene2idx[px0]
            gene_dep_mat[px0_idx, x_idx] = 1
    gene_con_mat = 1 * (gene_dep_mat + gene_dep_mat.T) > 0

    return gene_par, gene_mol, gene2idx, idx2gene, gene_dep_mat, gene_con_mat


def molcon_to_genecon(mol_con_est, idx2mol, gene2idx, thr=1e-4):
    # n_mol = len(idx2mol)
    # mol_con_est = 1 * (np.random.rand(n_mol, n_mol) > 0.99)

    n_gene = len(gene2idx)
    gene_con_est = np.zeros((n_gene, n_gene))
    m0_idx_lst, m1_idx_lst = np.where(np.abs(mol_con_est) > thr)
    for i in range(len(m0_idx_lst)):
        m0_idx = m0_idx_lst[i]
        m1_idx = m1_idx_lst[i]
        m0 = idx2mol[m0_idx]
        m1 = idx2mol[m1_idx]
        g0 = m0[:-5]
        g1 = m1[:-5]
        g0_idx = gene2idx[g0]
        g1_idx = gene2idx[g1]
        gene_con_est[g0_idx, g1_idx] = 1
        gene_con_est[g1_idx, g0_idx] = 1

    return gene_con_est


def make_iddn_dep_prior(
    mol_layer, mol2idx, allow_mrna_pairs=False, allow_protein_pairs=False
):
    # iDDN dependency constraints
    n_mol = len(mol_layer)
    dep_mat_prior = np.zeros((n_mol, n_mol))
    dep_mat_prior_loose = np.zeros((n_mol, n_mol))

    mol_names = list(mol_layer.keys())
    mol_type = []
    for m in mol_names:
        if "gene" in m:
            if m.endswith("_mrna"):
                mol_type.append("gene mrna")
            else:
                mol_type.append("gene prot")
        else:
            if m.endswith("_mrna"):
                mol_type.append("tf mrna")
            else:
                mol_type.append("tf prot")

    for i0, m0 in enumerate(mol_names):
        for i1, m1 in enumerate(mol_names):
            # cognate pairs
            i0x = mol2idx[m0]
            i1x = mol2idx[m1]
            if m0[:-5] == m1[:-5]:
                if m0.endswith("_mrna"):
                    dep_mat_prior[i0x, i1x] = 1
                else:
                    dep_mat_prior[i1x, i0x] = 1

            if mol_type[i0] == "gene mrna" and mol_type[i1] == "tf prot":
                dep_mat_prior[i1x, i0x] = 1
            if mol_type[i0] == "tf mrna" and mol_type[i1] == "tf prot":
                dep_mat_prior[i1x, i0x] = 1

            # The loose version do not have knowledge of TFs
            if mol_type[i0] == "tf mrna" or mol_type[i0] == "gene mrna":
                if mol_type[i1] == "gene prot" or mol_type[i1] == "tf prot":
                    dep_mat_prior_loose[i1x, i0x] = 1

            if allow_mrna_pairs:
                if mol_type[i0] == "gene mrna" and mol_type[i1] == "tf mrna":
                    dep_mat_prior[i1x, i0x] = 1
            if allow_protein_pairs:
                if mol_type[i0] == "gene prot" and mol_type[i1] == "tf prot":
                    dep_mat_prior[i1x, i0x] = 1

    dep_mat_prior_loose[dep_mat_prior > 0] = 1

    return dep_mat_prior, dep_mat_prior_loose


def molmat_to_genemat(t1_lst, t2_lst, idx2mol, gene2idx):
    t1_lst_gene = []
    t2_lst_gene = []
    for t1 in t1_lst:
        t1_gene = molcon_to_genecon(t1, idx2mol, gene2idx)
        t1_lst_gene.append(t1_gene)
    for t2 in t2_lst:
        t2_gene = molcon_to_genecon(t2, idx2mol, gene2idx)
        t2_lst_gene.append(t2_gene)
    return t1_lst_gene, t2_lst_gene
