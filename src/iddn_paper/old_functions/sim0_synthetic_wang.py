import numpy as np
import networkx as nx


def barabasi_albert_digraph(n, m_min, m_max, n_input, rep_init=5):
    G = nx.DiGraph()

    # nodes with external inputs
    G.add_nodes_from(range(n_input))

    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = list(range(n_input)) * rep_init

    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Choose m0 unique nodes from the existing nodes
        # Each time we add from 1 to m edges
        # Pick uniformly from repeated_nodes (preferential attachment)
        m0 = np.random.randint(m_min, m_max + 1)
        if len(repeated_nodes) > m0:
            targets = np.unique(np.random.choice(repeated_nodes, m0, replace=False))
        else:
            targets = np.random.choice(repeated_nodes, 1, replace=False)

        # Add edges to m nodes from the source.
        G.add_edges_from(zip(targets, [source] * m0))

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)

        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m0)

        source += 1
    return G


def sim_steady_state_linear(
    mat_wt,
    n_sample,
):
    # for each node, find inputs and weights
    n_node = len(mat_wt)
    node_in_lst = []  # list of parents
    noise_scl_in_lst = []  # scale for noise
    dat_scl_in_lst = []  # scale for parent iddn_data
    for i in range(n_node):
        idx = np.where(mat_wt[:, i] != 0)[0]
        wt = mat_wt[:, i][idx]
        sign = np.ones_like(wt)
        sign[wt < 0] = -1
        # scl = np.sqrt((1 / wt) ** 2 - 1)  # Based on needed correlation
        scl = wt
        node_in_lst.append(idx)
        noise_scl_in_lst.append(scl)
        dat_scl_in_lst.append(sign)

    # run simulation
    dat = np.zeros((n_node, n_sample))
    stable_mask = np.zeros(n_node)  # All nodes got desired input
    n_max_steps = n_node * 2

    i = 0
    for i in range(n_max_steps):
        node_lst = np.random.permutation(n_node)
        if int(np.sum(stable_mask)) == n_node:
            break

        for node_cur in node_lst:
            if stable_mask[node_cur] == 1:
                continue

            if len(node_in_lst[node_cur]) == 0:
                # if this node has no input, use N(0,1)
                # then set it as stable and no longer update
                stable_mask[node_cur] = 1
                x = np.random.randn(n_sample)
            else:
                # if all inputs are already stable, set current node as stable
                if np.sum(stable_mask[node_in_lst[node_cur]] == 0) == 0:
                    stable_mask[node_cur] = 1

                # include the contribution of each input
                # even some parents not stable, we still generate something
                # this might be useful when working on graphs with cycles
                x = np.zeros(n_sample)
                for i, node in enumerate(node_in_lst[node_cur]):
                    wt_dat = dat_scl_in_lst[node_cur][i]
                    wt_noise = noise_scl_in_lst[node_cur][i]
                    x += dat[node] * wt_dat + np.random.randn(n_sample) * wt_noise

            x = x - np.mean(x)
            x = x / np.std(x)
            dat[node_cur] = x

    print(i, np.sum(stable_mask))
    dat = dat.T

    return dat


################################################################
# SynTReN like
################################################################


def sim_syntren_like(
    regu_edges,
    gene_val_lst,
    mrna_post_scale_lst,
    n_steps=10,
    noise_sigma=0.1,
    hill_coef=4,
    hill_thr=1,
):
    n_genes = len(gene_val_lst)

    # make sure the dynamic range from 0 to 2, even there is no repressor or activator
    node_regu_ofst = np.zeros(n_genes)
    node_regu_scale = np.ones(n_genes)

    for key, val in regu_edges.items():
        # n_act = np.sum(val[1] == 1)
        # n_rep = np.sum(val[1] == 0)
        # node_regu_scale[key] = 1/(0.5**n_rep * (1.5)**n_act)

        if np.sum(val[1] == 1) == 0:
            # all repressor
            node_regu_scale[key] = 2
        if np.sum(val[1] == 0) == 0:
            # all activator
            node_regu_ofst[key] = -1
            node_regu_scale[key] = 2

    prot_val_noisy = None
    for _ in range(n_steps):
        # protein production with randomness
        prot_temp = gene_val_lst * mrna_post_scale_lst
        noise = np.random.randn(n_genes) * noise_sigma  # FIXME: more realistic noise
        prot_val_noisy = prot_temp + noise
        prot_Val_hill = (prot_val_noisy / hill_thr) ** hill_coef

        # activator
        # NOTE: maybe use a smaller offset is there are multiple activators
        prot_act = 1 + prot_Val_hill / (1 + prot_Val_hill)
        # repressor
        prot_rep = 1 / (1 + prot_Val_hill)

        # regulation of mRNA levels
        for gene_idx, edge in regu_edges.items():
            n_regulators = len(edge[0])
            mrna_val = 1
            # if gene_idx==9:
            #     print('hi')
            for i in range(n_regulators):
                tf_idx = edge[0][i]
                sign = edge[1][i]
                if sign > 0:
                    tf_indi = prot_act[tf_idx]
                else:
                    tf_indi = prot_rep[tf_idx]
                mrna_val = mrna_val * tf_indi

            mrna_val = (mrna_val + node_regu_ofst[gene_idx]) * node_regu_scale[gene_idx]
            gene_val_lst[gene_idx] = mrna_val + np.random.randn() * 0.1 * mrna_val

    return gene_val_lst, prot_val_noisy
