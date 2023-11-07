from os import path
import numpy as np
from args import EnrichTask, PathwayResults
from statistic_methods import bh_correction
from utils import load_propagation_scores
from utils import read_network, load_pathways_genes
from visualization_tools import plot_enrichment_table
import gseapy as gp
from typing import List, Dict
import pandas as pd


def run_gsea(task, ranked_genes: Dict[str, float], gene_sets: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Run Gene Set Enrichment Analysis (GSEA) using the gseapy library.

    Parameters:
    - ranked_genes (Dict[str, float]): A dictionary of gene IDs mapped to their respective scores.
    - gene_sets (Dict[str, List[str]]): A dictionary where keys are pathway names and values are lists of gene IDs in that pathway.

    Returns:
    - Dict[str, float]: A dictionary where keys are pathway names and values are the enrichment p-values for those pathways.
    """

    # Convert dictionary to Series, ensuring it's of float data type
    ranked_gene_series = pd.Series(ranked_genes, dtype=float)

    ranked_gene_series.index = ranked_gene_series.index.astype(str)
    gene_sets = {str(k): [str(g) for g in v] for k, v in gene_sets.items()}

    # Run GSEA
    enrichment_results = gp.prerank(rnk=ranked_gene_series, gene_sets=gene_sets, min_size=1, max_size=5000, outdir=None,
                                    no_plot=True)
    results_df = pd.DataFrame(enrichment_results.res2d)
    # save to csv
    results_df.to_csv('gsea_results.csv')

    # Extract and return the p-values for each pathway
    for index, row in results_df.iterrows():
        p_value = row['NOM p-val']
        adj_p_value = row['FDR q-val']
        direction = row['ES'] > 0
        # add to task results by pathway
        task.results[row['Term']] = PathwayResults(p_value=p_value, direction=direction, adj_p_value=adj_p_value)


def get_scores(task):
    """
    takes a task, a network graph and a general args object and returns a dictionary of scores
    :param task:
    :return:
    """

    if isinstance(task, EnrichTask):
        result_dict = load_propagation_scores(task, propagation_file_name=task.propagation_file)
        gene_id_to_idx = {xx: x for x, xx in result_dict['gene_idx_to_id'].items()}
        scores = result_dict[task.target_field]
        if task.constrain_to_experiment_genes:
            gene_id_to_idx = {id: idx for id, idx in gene_id_to_idx.items() if id in result_dict['propagation_input']}
        scores = {id: scores[idx][0] for id, idx in gene_id_to_idx.items()}
    else:
        raise ValueError('Invalid task')
    return scores


def load_network_and_pathways(general_args):
    network_graph = read_network(general_args.network_file_path)
    genes_by_pathway = load_pathways_genes(general_args.pathway_members_path)
    interesting_pathways = list(genes_by_pathway.keys())

    return network_graph, interesting_pathways, genes_by_pathway


def process_tasks(task_list, network_graph, general_args, interesting_pathways, genes_by_pathway):
    pathways_to_display = set()
    all_genes_by_pathway_filtered = {}

    # Create a set to hold all genes that are in some pathway and are in the network.
    all_genes_in_filtered_pathways_and_network = set()

    for task in task_list:
        scores = get_scores(task)

        # Filter genes for each pathway, contains only genes that are in the experiment and in the pathway file
        genes_by_pathway_filtered = {pathway: [id for id in genes_by_pathway[pathway] if id in scores]
                                     for pathway in interesting_pathways}

        # keep only pathway with certain amount of genes
        pathways_with_many_genes = [pathway_name for pathway_name in genes_by_pathway_filtered.keys() if
                                    (len(genes_by_pathway_filtered[
                                             pathway_name]) >= general_args.minimum_gene_per_pathway and len(
                                        genes_by_pathway_filtered[
                                            pathway_name]) <= general_args.maximum_gene_per_pathway)]
        # manually add REACTOME_NEUROTRANSMITTER_RECEPTORS_AND_POSTSYNAPTIC_SIGNAL_TRANSMISSION'
        pathways_with_many_genes.append('REACTOME_NEUROTRANSMITTER_RECEPTORS_AND_POSTSYNAPTIC_SIGNAL_TRANSMISSION')

        # Update all_genes_in_filtered_pathways_and_network
        for pathway in pathways_with_many_genes:
            all_genes_in_filtered_pathways_and_network.update(genes_by_pathway_filtered[pathway])

        # Intersect with network nodes to refine the set
        all_genes_in_filtered_pathways_and_network &= set(network_graph.nodes)

        for pathway in pathways_with_many_genes:
            pathways_to_display.add(pathway)

        if general_args.use_gsea:  # Assume you add this flag to general_args or similar
            ranked_genes = get_scores(task)
            # Replace with how you get your ranked genes
            run_gsea(task, ranked_genes, genes_by_pathway)
        else:
            # Perform statistical tests
            print('after filtering', len(pathways_with_many_genes))
            for pathway in pathways_with_many_genes:
                pathway_scores = [scores[id] for id in genes_by_pathway_filtered[pathway]]
                # background_scores1 = [scores[id] for id in scores.keys() if id not in genes_by_pathway_filtered[pathway]]
                background_scores = [scores[id] for id in all_genes_in_filtered_pathways_and_network if
                                     id not in genes_by_pathway_filtered[pathway]]
                result = task.statistic_test(pathway_scores, background_scores)
                task.results[pathway] = PathwayResults(p_value=result.p_value, direction=result.directionality)

        # Store the filtered genes by pathway for this task
        all_genes_by_pathway_filtered[task.name] = genes_by_pathway_filtered

    pathways_to_display = np.sort(list(pathways_to_display))

    return pathways_to_display, all_genes_by_pathway_filtered, pathways_with_many_genes


# def create_matrices(genes_by_pathway, task_list, pathways_to_display):
#     pathways_to_display = np.sort(list(pathways_to_display))
#     p_vals_mat = np.ones((len(pathways_to_display), len(task_list)))
#     adj_p_vals_mat = np.ones_like(p_vals_mat)
#     directions_mat = np.zeros_like(p_vals_mat)
#     coll_names_in_heatmap = []
#     aaaa_geneset = set(genes_by_pathway['decoy']['AAAA_PATHWAY'])
#
#     for t, task in enumerate(task_list):
#         indexes = []
#         for p, pathway in enumerate(pathways_to_display):
#             if pathway in task.results:
#                 indexes.append(p)
#                 p_vals_mat[p, t] = task.results[pathway].p_value
#                 directions_mat[p, t] = task.results[pathway].direction
#         adj_p_vals_mat[indexes, t] = bh_correction(p_vals_mat[indexes, t])
#         coll_names_in_heatmap.append(task.name)
#
#         # Find pathways with min and max p-value
#         min_pval_pathway = pathways_to_display[np.argmin(p_vals_mat[:, t])]
#         max_pval_pathway = pathways_to_display[np.argmax(p_vals_mat[:, t])]
#
#         # Find pathways with min and max adjusted p-value
#         min_adjpval_pathway = pathways_to_display[np.argmin(adj_p_vals_mat[:, t])]
#         max_adjpval_pathway = pathways_to_display[np.argmax(adj_p_vals_mat[:, t])]
#
#         # Get gene sets for these pathways
#         min_pval_geneset = set(genes_by_pathway['decoy'][min_pval_pathway])
#         max_pval_geneset = set(genes_by_pathway['decoy'][max_pval_pathway])
#         min_adjpval_geneset = set(genes_by_pathway['decoy'][min_adjpval_pathway])
#         max_adjpval_geneset = set(genes_by_pathway['decoy'][max_adjpval_pathway])
#
#         # Calculate percent of shared genes
#         shared_with_min_pval = len(aaaa_geneset.intersection(min_pval_geneset)) / len(aaaa_geneset) * 100
#         shared_with_max_pval = len(aaaa_geneset.intersection(max_pval_geneset)) / len(aaaa_geneset) * 100
#         shared_with_min_adjpval = len(aaaa_geneset.intersection(min_adjpval_geneset)) / len(aaaa_geneset) * 100
#         shared_with_max_adjpval = len(aaaa_geneset.intersection(max_adjpval_geneset)) / len(aaaa_geneset) * 100
#
#         print(f'Percent of genes shared between AAAA_PATHWAY and {min_pval_pathway} (min p-value): {shared_with_min_pval:.2f}%')
#         print(f'Percent of genes shared between AAAA_PATHWAY and {max_pval_pathway} (max p-value): {shared_with_max_pval:.2f}%')
#         print(f'Percent of genes shared between AAAA_PATHWAY and {min_adjpval_pathway} (min adj. p-value): {shared_with_min_adjpval:.2f}%')
#         print(f'Percent of genes shared between AAAA_PATHWAY and {max_adjpval_pathway} (max adj. p-value): {shared_with_max_adjpval:.2f}%')
#
#     return p_vals_mat, adj_p_vals_mat, directions_mat, coll_names_in_heatmap


def process_matrices(task_list, pathways_to_display, general_args):
    pathways_to_display = np.sort(list(pathways_to_display))
    p_vals_mat = np.ones((len(pathways_to_display), len(task_list)))
    adj_p_vals_mat = np.ones_like(p_vals_mat)
    directions_mat = np.zeros_like(p_vals_mat)
    coll_names_in_heatmap = []

    # Create matrices
    for t, task in enumerate(task_list):
        indexes = []
        for p, pathway in enumerate(pathways_to_display):
            if pathway in task.results:
                indexes.append(p)
                p_vals_mat[p, t] = task.results[pathway].p_value
                directions_mat[p, t] = task.results[pathway].direction
        adj_p_vals_mat[indexes, t] = bh_correction(p_vals_mat[indexes, t])
        coll_names_in_heatmap.append(task.name)

    # Filter and adjust matrices
    keep_rows = np.nonzero(np.any(adj_p_vals_mat <= general_args.significant_pathway_threshold, axis=1))[0]
    pathways_to_display = list(pathways_to_display)  # Convert set to list
    pathways_to_display = [pathways_to_display[x] for x in keep_rows]  # Now this should work
    specific_pathways = [
        "REACTOME_PRESYNAPTIC_DEPOLARIZATION_AND_CALCIUM_CHANNEL_OPENING",
        "REACTOME_NEUROTRANSMITTER_RECEPTORS_AND_POSTSYNAPTIC_SIGNAL_TRANSMISSION",
        "REACTOME_DOPAMINE_CLEARANCE_FROM_THE_SYNAPTIC_CLEFT",
        "REACTOME_ACTIVATION_OF_NMDA_RECEPTORS_AND_POSTSYNAPTIC_EVENTS",
        "REACTOME_PRESYNAPTIC_FUNCTION_OF_KAINATE_RECEPTORS",
        "REACTOME_HIGHLY_SODIUM_PERMEABLE_POSTSYNAPTIC_ACETYLCHOLINE_NICOTINIC_RECEPTORS",
        "REACTOME_HIGHLY_CALCIUM_PERMEABLE_POSTSYNAPTIC_NICOTINIC_ACETYLCHOLINE_RECEPTORS",
        "REACTOME_SYNAPTIC_ADHESION_LIKE_MOLECULES",
        "WP_SPLICING_FACTOR_NOVA_REGULATED_SYNAPTIC_PROTEINS",
        "WP_DISRUPTION_OF_POSTSYNAPTIC_SIGNALING_BY_CNV",
        "WP_SYNAPTIC_VESICLE_PATHWAY",
        "WP_SYNAPTIC_SIGNALING_PATHWAYS_ASSOCIATED_WITH_AUTISM_SPECTRUM_DISORDER",
        "KEGG_LYSOSOME",
        "REACTOME_LYSOSPHINGOLIPID_AND_LPA_RECEPTORS",
        "REACTOME_LYSOSOME_VESICLE_BIOGENESIS",
        "REACTOME_PREVENTION_OF_PHAGOSOMAL_LYSOSOMAL_FUSION",
        "PID_LYSOPHOSPHOLIPID_PATHWAY",
        "BIOCARTA_CARM_ER_PATHWAY",
        "REACTOME_SYNTHESIS_OF_PIPS_AT_THE_ER_MEMBRANE",
        "REACTOME_ER_TO_GOLGI_ANTEROGRADE_TRANSPORT",
        "REACTOME_N_GLYCAN_TRIMMING_IN_THE_ER_AND_CALNEXIN_CALRETICULIN_CYCLE",
        "REACTOME_COPI_DEPENDENT_GOLGI_TO_ER_RETROGRADE_TRAFFIC",
        "REACTOME_COPI_INDEPENDENT_GOLGI_TO_ER_RETROGRADE_TRAFFIC",
        "REACTOME_INTRA_GOLGI_AND_RETROGRADE_GOLGI_TO_ER_TRAFFIC",
        "REACTOME_GOLGI_TO_ER_RETROGRADE_TRANSPORT",
        "REACTOME_ER_QUALITY_CONTROL_COMPARTMENT_ERQC",
        "WP_METABOLISM_OF_SPHINGOLIPIDS_IN_ER_AND_GOLGI_APPARATUS",
        "PID_ER_NONGENOMIC_PATHWAY",
        "WP_NEUROINFLAMMATION_AND_GLUTAMATERGIC_SIGNALING",
        "WP_RELATIONSHIP_BETWEEN_INFLAMMATION_COX2_AND_EGFR",
        "WP_RESISTIN_AS_A_REGULATOR_OF_INFLAMMATION",
        "WP_APOE_AND_MIR146_IN_INFLAMMATION_AND_ATHEROSCLEROSIS",
        "WP_SUPRESSION_OF_HMGB1_MEDIATED_INFLAMMATION_BY_THBD",
        "WP_RESOLVIN_E1_AND_RESOLVIN_D1_SIGNALING_PATHWAYS_PROMOTING_INFLAMMATION_RESOLUTION",
        "WP_NEUROINFLAMMATION"
    ]  # List of specific pathways
    # print specific pathways if they are in the list
    for pathway in specific_pathways:
        if pathway in pathways_to_display:
            print(pathway)
    print('DONE PRINTING PATHWAYS')
    adj_p_vals_mat = adj_p_vals_mat[keep_rows, :]
    directions_mat = directions_mat[keep_rows, :]

    return adj_p_vals_mat, directions_mat, pathways_to_display, coll_names_in_heatmap


def prepare_row_names(pathways_to_display, genes_by_pathway_filtered):
    row_names = ['({}) {}'.format(len(genes_by_pathway_filtered[pathway]), pathway) for pathway in pathways_to_display]
    row_names = [x.replace("-", "-\n") for x in row_names]
    return row_names


def filter_by_minimum_p_values(p_vals_mat, adj_p_vals_mat, directions_mat, pathways_to_display, general_args):
    candidates = np.min(p_vals_mat, axis=1)
    ind = np.sort(
        np.argpartition(candidates, general_args.maximum_number_of_pathways)[:general_args.maximum_number_of_pathways])
    p_vals_mat = p_vals_mat[ind, :]
    adj_p_vals_mat = adj_p_vals_mat[ind, :]
    directions_mat = directions_mat[ind, :]
    pathways_to_display = [pathways_to_display[x] for x in ind]
    return p_vals_mat, adj_p_vals_mat, directions_mat, pathways_to_display


def plot_results(adj_p_vals_mat, directions_mat, row_names, general_args, coll_names_in_heatmap,
                 dataset_type, n_pathways_before):
    res = -np.log10(adj_p_vals_mat)
    fig_out_dir = path.join(general_args.output_path, general_args.figure_name)
    plot_enrichment_table(res, directions_mat, row_names, fig_out_dir, experiment_names=coll_names_in_heatmap,
                          title=general_args.figure_title + ' {} {}/{}'.format(dataset_type,
                                                                               len(row_names), n_pathways_before),
                          res_type='-log10(p_val)', adj_p_value_threshold=general_args.significant_pathway_threshold)


def run(task_list, general_args, dataset_type=''):
    """
    takes a list of tasks, a general args object and an optional dataset type and runs the pathway enrichment analysis
    :param task_list: list of tasks
    :param general_args: general args object
    :param dataset_type: optional dataset type
    :return: None
    """
    network_graph, interesting_pathways, genes_by_pathway = load_network_and_pathways(general_args)

    pathways_to_display, genes_by_pathway_filtered, pathways_with_many_genes = (process_tasks(task_list, network_graph,
                                                                                              general_args,
                                                                                              interesting_pathways,
                                                                                              genes_by_pathway))

    # Create matrices for p-values, adjusted p-values, and directions
    adj_p_vals_mat, directions_mat, pathways_to_display, coll_names_in_heatmap = process_matrices(task_list,
                                                                                                  pathways_to_display,
                                                                                                  general_args)

    row_names = ['{}'.format(pathway.replace("_", " ")) for pathway in pathways_to_display]

    # Plot the results
    plot_results(adj_p_vals_mat, directions_mat, row_names, general_args, coll_names_in_heatmap,
                 dataset_type, len(pathways_with_many_genes))