import numpy as np


def create_gene_network(
    n_hub_regulator=1,
    n_secondary_regulator=7,
    n_other_genes=10,
    wt_top_regulator=3,
):
    # We begin with a gene network, then translate to molecular level
    # A top regulator is treated as external input
    # Second regulators are regulated by other regulators, include top and second ones
    # Other genes are protein coding genes that are not TFs

    n_regulators = n_hub_regulator + n_secondary_regulator
    regulator_idx = np.arange(n_regulators)

    regulator_weight = np.ones(n_regulators)
    regulator_weight[:n_hub_regulator] = wt_top_regulator

    n_gene = n_regulators + n_other_genes

    gene_type = dict()
    gene_parent = dict()
    for i in range(n_gene):
        gene_parent[i] = []

        if i < n_hub_regulator:
            gene_type[i] = "hub_reg"
        elif i < n_regulators:
            gene_type[i] = "reg"
        else:
            gene_type[i] = "other"

        if gene_type[i] == "reg":
            parent_candidates = regulator_idx[:i]
            parent_candidates_wt = regulator_weight[:i]
            thr2 = 0.0  # 0.5
        elif gene_type[i] == "other":
            parent_candidates = regulator_idx[:n_regulators]
            parent_candidates_wt = regulator_weight[:n_regulators]
            thr2 = 0.0  # 0.5
        else:
            # Assume all hubs are independent
            continue

        # choose one or two regulators
        if len(parent_candidates) == 0:
            continue
        elif len(parent_candidates) == 1:
            # If there is only one possible parent, choose that one.
            gene_parent[i].extend(parent_candidates)
            regulator_weight[parent_candidates] += 1
        else:
            parent_candidates_wt = parent_candidates_wt / np.sum(parent_candidates_wt)
            p = np.random.choice(
                parent_candidates, size=2, replace=False, p=parent_candidates_wt
            ).tolist()
            if np.random.rand() < thr2:
                gene_parent[i].extend(p)
                regulator_weight[p[0]] += 1
                regulator_weight[p[1]] += 1
            else:
                gene_parent[i].append(p[0])
                regulator_weight[p[0]] += 1

    return gene_type, gene_parent, regulator_weight


def create_mol_network(gene_type, gene_parent):
    # molecular network for TF protein and mRNA
    # Feedback not used here
    # molecule species (level in the ODE). 0: RNA, 1: protein, 2: miRNA
    # molecule type: mRNA, miRNA, TFmRNA, lncRNA, protein, etc.
    # role of a parent: 0: TF activate, 1: TF repress, 2: dosage in translation, 3: miRNA repress

    # All nodes
    mol_type = dict()  # not needed for final ODE, only used for building network
    mol_layer = dict()
    n_gene = len(gene_type)
    for i in range(n_gene):
        if gene_type[i] == "hub_reg":
            mol_layer[f"rna_{i}"] = 0
            mol_layer[f"prot_{i}"] = 1
            mol_type[f"rna_{i}"] = "TFmRNA"
            mol_type[f"prot_{i}"] = "TFprotein"
        elif gene_type[i] == "reg":
            if np.random.rand() < 0.0:  # 0.5
                # Assign some regulators as miRNAs
                mol_layer[f"rna_{i}"] = 2
                mol_type[f"rna_{i}"] = "miRNA"
            else:
                mol_layer[f"rna_{i}"] = 0
                mol_layer[f"prot_{i}"] = 1
                mol_type[f"rna_{i}"] = "TFmRNA"
                mol_type[f"prot_{i}"] = "TFprotein"
        else:
            mol_layer[f"rna_{i}"] = 0
            mol_layer[f"prot_{i}"] = 1
            mol_type[f"rna_{i}"] = "mRNA"
            mol_type[f"prot_{i}"] = "protein"

    # All edges
    mol_parent = dict()
    mol_parent_roles = dict()
    for x in mol_layer.keys():
        mol_parent[x] = []
        mol_parent_roles[x] = []

    for x, px_lst in gene_parent.items():
        # RNA to protein if x is a protein coding gene
        if f"prot_{x}" in mol_layer:
            mol_parent[f"prot_{x}"].append(f"rna_{x}")
            mol_parent_roles[f"prot_{x}"].append(2)

        # edges from regulator to x
        for px in px_lst:
            if mol_type[f"rna_{px}"] == "miRNA":
                if f"prot_{x}" in mol_layer:
                    # parent is miRNA. We assume miRNA only repress the protein expression
                    mol_parent[f"prot_{x}"].append(f"rna_{px}")
                    mol_parent_roles[f"prot_{x}"].append(3)
            else:
                # parent is TF, randomly assign a role (activator or repressor)
                mol_parent[f"rna_{x}"].append(f"prot_{px}")
                mol_parent_roles[f"rna_{x}"].append(np.random.randint(2))

    return mol_layer, mol_parent, mol_parent_roles, mol_type


def make_two_conditions_mol_net(mol_parent, mol_parent_roles, ratio=0.25):
    # Node is not changed, just remove edges
    # Assume the ratio parameter is 25%
    # For each regulatory edge, 25% chance to condition 1 only, 25% chance to condition 2 only
    # 50% chance to both.

    mol_parent1 = dict()
    mol_parent2 = dict()
    mol_parent_roles1 = dict()
    mol_parent_roles2 = dict()

    for mol_name in mol_parent.keys():
        par_now = mol_parent[mol_name]
        par_role_now = mol_parent_roles[mol_name]

        mol_parent1[mol_name] = []
        mol_parent2[mol_name] = []
        mol_parent_roles1[mol_name] = []
        mol_parent_roles2[mol_name] = []

        for i in range(len(par_now)):
            xx = np.random.rand()

            # if (mol_parent_roles[mol_name][i] == 2) or (xx > 2 * ratio):
            if xx > 2 * ratio:
                # assign to both conditions
                # TODO: ignore or allow the translation edge?
                mol_parent1[mol_name].append(par_now[i])
                mol_parent2[mol_name].append(par_now[i])
                mol_parent_roles1[mol_name].append(par_role_now[i])
                mol_parent_roles2[mol_name].append(par_role_now[i])
                continue

            if xx < ratio:
                # assign to condition 1
                mol_parent1[mol_name].append(par_now[i])
                mol_parent_roles1[mol_name].append(par_role_now[i])
            elif xx < 2 * ratio:
                # assign to condition 2
                mol_parent2[mol_name].append(par_now[i])
                mol_parent_roles2[mol_name].append(par_role_now[i])
            else:
                raise "Something wrong"

    return mol_parent1, mol_parent2, mol_parent_roles1, mol_parent_roles2


def mol_network_to_index(mol_layer, mol_parent, mol_parent_roles):
    # change molecular names to indices

    # Index the molecules by layers, for iDINGO and maybe for visualization
    layer_count = []
    n_max_layer = 10
    mol_to_idx = dict()
    idx_to_mol = dict()
    layer_idx = np.array(list(mol_layer.values()))
    i = 0
    for n in range(n_max_layer):
        layer_count.append(np.sum(layer_idx == n))
        for x in mol_layer.keys():
            if mol_layer[x] == n:
                mol_to_idx[x] = i
                idx_to_mol[i] = x
                i = i + 1

    idx_layer = dict()
    idx_parent = dict()
    idx_parent_roles = dict()
    for x, px in mol_layer.items():
        idx_layer[mol_to_idx[x]] = px

    for x, px in mol_parent.items():
        idx_parent[mol_to_idx[x]] = [mol_to_idx[px0] for px0 in px]

    for x, px in mol_parent_roles.items():
        idx_parent_roles[mol_to_idx[x]] = px

    mol_to_idx = dict(sorted(mol_to_idx.items()))
    idx_to_mol = dict(sorted(idx_to_mol.items()))

    idx_layer = dict(sorted(idx_layer.items()))
    idx_parent = dict(sorted(idx_parent.items()))
    idx_parent_roles = dict(sorted(idx_parent_roles.items()))

    return idx_layer, idx_parent, idx_parent_roles, mol_to_idx, idx_to_mol, layer_count


def make_iddn_dep_prior(mol_type, mol2idx):
    # iDDN dependency constraints
    # mRNA, protein, TFmRNA, TFprotein, miRNA

    n_mol = len(mol_type)
    dep_mat_prior = np.zeros((n_mol, n_mol))
    dep_mat_prior_loose = np.zeros((n_mol, n_mol)) + 1
    # mol_names = list(mol_type.keys())

    for m0 in mol_type.keys():
        for m1 in mol_type.keys():
            # Translation
            i0x = mol2idx[m0]
            i1x = mol2idx[m1]
            gene_mol0, gene_idx0 = m0.split("_")
            # print(gene_mol0, gene_idx0)
            _, gene_idx1 = m1.split("_")
            if gene_idx0 == gene_idx1:
                if gene_mol0 == "rna":
                    dep_mat_prior[i0x, i1x] = 1
                else:
                    dep_mat_prior[i1x, i0x] = 1

            # TFprotein - mRNA/TFmRNA/miRNA
            if mol_type[m1] == "TFprotein":
                if mol_type[m0] in ("mRNA", "TFmRNA", "miRNA"):
                    dep_mat_prior[i1x, i0x] = 1

            if mol_type[m1] in ("protein", "TFprotein"):
                if mol_type[m0] in ("mRNA", "TFmRNA", "miRNA"):
                    dep_mat_prior_loose[i1x, i0x] = 1

            # miRNA - protein/TFprotein
            if mol_type[m1] == "miRNA":
                if mol_type[m0] in ("protein", "TFprotein"):
                    dep_mat_prior[i1x, i0x] = 1

    dep_mat_prior_loose = dep_mat_prior_loose + dep_mat_prior

    return dep_mat_prior, dep_mat_prior_loose


def get_dep_mat(idx_parent):
    n_node = len(idx_parent)
    dep_mat = np.zeros((n_node, n_node))
    for x, px in idx_parent.items():
        dep_mat[px, x] = 1

    con_mat = dep_mat + dep_mat.T
    con_mat[con_mat > 1] = 1
    return dep_mat, con_mat


def get_translation_mat(idx_parent, idx_parent_roles):
    n_node = len(idx_parent)
    translation_mat = np.zeros((n_node, n_node))
    for x in range(n_node):
        px = idx_parent[x]
        px_roles = idx_parent_roles[x]
        for i in range(len(px)):
            if px_roles[i] == 2:
                translation_mat[px[i], x] = 1
                translation_mat[x, px[i]] = 1

    return translation_mat


def prep_net_for_sim(mol_layer, mol_par, mol_par_roles, mol_type):
    idx_layer, idx_par, idx_par_roles, mol2idx, idx2mol, layer_cnt = (
        mol_network_to_index(mol_layer, mol_par, mol_par_roles)
    )
    net_info = dict(
        mol_layer=mol_layer,
        mol_par=mol_par,
        mol_par_roles=mol_par_roles,
        mol_type=mol_type,
        idx_layer=idx_layer,
        idx_par=idx_par,
        idx_par_roles=idx_par_roles,
        mol2idx=mol2idx,
        idx2mol=idx2mol,
        layer_count=layer_cnt,
    )
    dep_mat, con_mat = get_dep_mat(idx_par)
    return net_info, dep_mat, con_mat


# def make_mol_network(
#     n_top_regulator=3,
#     n_second_regulator=27,
#     n_other_genes=70,
#     wt_top_regulator=3,
#     two_condition_ratio=0.25,
# ):
#     # Make a gene network
#     gene_type, gene_par = create_gene_network(
#         n_hub_regulator=n_top_regulator,
#         n_secondary_regulator=n_second_regulator,
#         n_other_genes=n_other_genes,
#         wt_top_regulator=wt_top_regulator,
#     )
#
#     # Make a molecule level network
#     mol_layer, mol_par, mol_par_roles, mol_type = create_mol_network(
#         gene_type,
#         gene_par,
#     )
#
#     # Make two conditions by removing some edges in each condition
#     mol_par1, mol_par2, mol_par_roles1, mol_par_roles2 = make_two_conditions_mol_net(
#         mol_par,
#         mol_par_roles,
#         ratio=two_condition_ratio,
#     )
#
#     # Simulation for each condition
#     net_info1, dep_mat1, con_mat1 = prep_net_for_sim(
#         mol_layer, mol_par1, mol_par_roles1, mol_type
#     )
#     net_info2, dep_mat2, con_mat2 = prep_net_for_sim(
#         mol_layer, mol_par2, mol_par_roles2, mol_type
#     )
#
#     return net_info1, net_info2, dep_mat1, dep_mat2, con_mat1, con_mat2
