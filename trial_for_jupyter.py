import pandas as pd
import numpy as np
from os import path, listdir, makedirs
from scipy.stats import rankdata, ranksums
import matplotlib.pyplot as plt
import shutil


MIN_GENE_PER_PATHWAY = 20
MAX_GENE_PER_PATHWAY = 200
FDR_THRESHOLD = 0.05
JAC_THRESHOLD = 0.2
P_VALUE_THRESHOLD = 0.05
root_path = '/mnt/c/Users/pickh/PycharmProjects/Yair_propagation'
output_path = path.join(root_path, 'Outputs')
Experiment_name = 'Parkinson'
data_file = 'Data'
species = 'H_sapiens'
genes_names_file = 'H_sapiens.gene_info'
pathway_file = 'pathways'
data_dir = path.join(root_path, data_file)
input_dir = path.join(root_path, 'Inputs', 'experiments_data', Experiment_name)
genes_names_file_path = path.join(data_dir, species, 'genes_names', genes_names_file)
pathway_file_dir = path.join(data_dir, species, 'pathways', pathway_file)

temp_output_folder = path.join(root_path, 'Outputs', 'Temp')
# Ensure output directories exist
makedirs(temp_output_folder, exist_ok=True)

def load_pathways_genes():
    pathways = {}
    with open(pathway_file_dir, 'r') as f:
        for line in f:
            parts = line.strip().upper().split('\t')  # Split each line into parts
            if len(parts) < 3 or not parts[1].isdigit():
                continue

            pathway_name, pathway_size = parts[0], int(parts[1])

            genes = [int(gene) for gene in parts[2].split()[:pathway_size] if gene.isdigit()]
            pathways[pathway_name] = genes
    return pathways


def get_scores(test_name):
    # Path to the file containing the raw scores (adjust as necessary)
    raw_scores_file_path = path.join(input_dir, f'{test_name}.csv')

    try:
        # Load raw data from the file
        raw_data = pd.read_csv(raw_scores_file_path, usecols=['GeneID', 'Score'])
        # Assuming 'GeneID' and 'Score' are columns in the raw data
        raw_data.sort_values(by='GeneID', inplace=True)

        # Create a dictionary for gene_id_to_score using vectorized operations
        scores_dict = raw_data.set_index('GeneID')['Score'].to_dict()
        return scores_dict

    except FileNotFoundError:
        print(f"File not found: {raw_scores_file_path}")
        return pd.DataFrame(), {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), {}


def bh_correction(p_values):
    p_vals_rank = rankdata(p_values, 'max') - 1
    p_vals_rank_ord = rankdata(p_values, 'ordinal') - 1

    p_values_sorted = np.zeros_like(p_vals_rank)
    p_values_sorted[p_vals_rank_ord] = np.arange(len(p_vals_rank_ord))

    p_vals = p_values * (len(p_values) / (p_vals_rank + 1))
    adj_p_vals_by_rank = p_vals[p_values_sorted]

    p_vals_ordered = np.minimum(adj_p_vals_by_rank, np.minimum.accumulate(adj_p_vals_by_rank[::-1])[::-1])
    adj_p_vals = p_vals_ordered[p_vals_rank]
    return adj_p_vals


def perform_statist(test_name):
    scores = get_scores(test_name)

    scores_keys = set(scores.keys())
    filtered_genes_by_pathway = {pathway: set(genes).intersection(scores_keys)
                                 for pathway, genes in genes_by_pathway.items()}

    pathways_with_many_genes = {pathway: genes for pathway, genes in filtered_genes_by_pathway.items()
                                if MIN_GENE_PER_PATHWAY <= len(genes) <= MAX_GENE_PER_PATHWAY}

    ks_p_values = []

    # Perform statistical tests
    for pathway, genes in pathways_with_many_genes.items():
        pathway_scores = [scores[gene_id] for gene_id in genes]
        background_scores = [scores[gene_id] for gene_id in scores_keys - genes]
        ks_p_values.append(kolmogorov_smirnov_test(pathway_scores, background_scores))

    # Apply BH correction
    adjusted_p_values = bh_correction(np.array(ks_p_values))

    significant_pathways_with_genes = {
        pathway: (pathways_with_many_genes[pathway], adjusted_p_values[i])
        for i, pathway in enumerate(pathways_with_many_genes)
        if adjusted_p_values[i] < 0.05
    }

    return significant_pathways_with_genes, scores

def perform_statist_mann_whitney(passed_ks_pathway_dict, scores):
    mw_p_values = []
    significant_pathways_with_genes = {}

    scores_keys = set(scores.keys())

    # Mann-Whitney U test and FDR
    for pathway, genes_info in passed_ks_pathway_dict.items():
        pathway_genes = set(genes_info[0])
        pathway_scores = [scores[gene_id] for gene_id in pathway_genes]
        background_scores = [scores[gene_id] for gene_id in scores_keys - pathway_genes]

        # Perform Mann-Whitney U Test
        mw_pval = wilcoxon_rank_sums_test(pathway_scores, background_scores)
        mw_p_values.append(mw_pval)


    # Apply BH correction to Mann-Whitney p-values
    adjusted_mw_p_values = bh_correction(np.array(mw_p_values))

    filtered_pathways = []
    for i, (pathway, genes) in enumerate(passed_ks_pathway_dict.items()):
        if adjusted_mw_p_values[i] < 0.05:  # Significance threshold
            significant_pathways_with_genes[pathway] = genes
            filtered_pathways.append({
                'Pathway': pathway,
                'Adjusted_p_value': adjusted_mw_p_values[i],
                'Genes': genes[0]
            })

    # Convert filtered pathways to a DataFrame
    pathways_df = pd.DataFrame(filtered_pathways)
    pathways_df.sort_values(by='Adjusted_p_value', inplace=True)

    # Filter pathways based on Jaccard index
    filtered_pathways = {}
    for i, row in pathways_df.iterrows():
        current_genes = set(row['Genes'])
        if not any(jaccard_index(current_genes, set(filtered_row['Genes'])) > JAC_THRESHOLD
                   for filtered_row in filtered_pathways.values()):
            filtered_pathways[row['Pathway']] = row

    return filtered_pathways


def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def kolmogorov_smirnov_test(experiment_scores, control_scores):
    # Convert lists to numpy arrays and sort
    experiment_scores = np.sort(experiment_scores)
    control_scores = np.sort(control_scores)
    # ks_stats, ks_p_value = ks_2samp(experiment_scores, control_scores, alternative='two-sided')
    # Initialize variables
    en1 = len(experiment_scores)
    en2 = len(control_scores)

    # Calculate empirical cumulative distribution functions for both sets
    data_all = np.concatenate([experiment_scores, control_scores])
    cdf_experiment = np.searchsorted(experiment_scores, data_all, side='right') / en1
    cdf_control = np.searchsorted(control_scores, data_all, side='right') / en2

    # Find the maximum distance
    D = np.max(np.abs(cdf_experiment - cdf_control))
    # Calculate the KS statistic
    en = np.sqrt(en1 * en2 / (en1 + en2))
    p_value = ks((en + 0.12 + 0.11 / en) * D)

    return p_value


def ks(alam):
    EPS1 = 1e-6  # Convergence criterion based on the term's absolute value
    EPS2 = 1e-10  # Convergence criterion based on the sum's relative value
    a2 = -2.0 * alam ** 2  # Squared and negated lambda for exponential calculation
    fac = 2.0
    sum = 0.0
    termbf = 0.0

    # Iteratively calculate the KS probability
    for j in range(1, 101):
        term = fac * np.exp(a2 * j ** 2)  # Calculate term of the series
        sum += term  # Add to sum

        # Check for convergence
        if np.abs(term) <= EPS1 * termbf or np.abs(term) <= EPS2 * sum:
            return sum

        fac = -fac  # Alternate the sign
        termbf = np.abs(term)  # Update term before flag

    # Return 1.0 if the series does not converge in 100 terms
    return 1.0


def wilcoxon_rank_sums_test(experiment_scores, elements_scores, alternative='two-sided'):
    p_vals = ranksums(experiment_scores, elements_scores, alternative=alternative).pvalue
    return p_vals


def print_enriched_pathways_to_file(filtered_pathways, output_folder, threshold=0.05):
    output_file_path = path.join(output_folder, f'{test_name}.txt')
    significant_count = 0

    with open(output_file_path, 'w') as file:
        for pathway, details in filtered_pathways.items():
            p_value = details.get('Adjusted_p_value')
            if p_value and p_value < threshold:
                file.write(f"{pathway} {p_value}\n")
                significant_count += 1



def read_scores(file_name):
    """Reads scores from a given file into a dictionary."""
    return pd.read_csv(file_name, sep=' ', header=None, names=['Pathway', 'Score'], index_col='Pathway')[
        'Score'].to_dict()


def read_pathways(file_name):
    """Reads pathways from a given file into a dictionary."""
    with open(file_name, 'r') as file:
        return {line.split()[0]: [int(gene) for gene in line.split()[2:]] for line in file}


def process_experiment(condition_file, experiment_file, pathways_file):
    """Processes an experiment and returns scores, enriched pathway genes, and mean scores."""
    enriched_pathway_dict = read_scores(condition_file)
    condition_data_df = pd.read_csv(experiment_file)
    experiment_data_filtered_df = condition_data_df[condition_data_df['Score'] != 0]

    pathways = read_pathways(pathways_file)
    enriched_pathway_genes = {}
    pathway_mean_scores = {}

    for pathway in enriched_pathway_dict.keys():
        pathway_genes = pathways[pathway]
        filtered_genes = experiment_data_filtered_df[experiment_data_filtered_df['GeneID'].isin(pathway_genes)]
        if not filtered_genes.empty:
            enriched_pathway_genes[pathway] = filtered_genes.set_index('GeneID')[
                ['Symbol', 'Score', 'P-value']].to_dict(
                orient='index')
            # # create array of genes with score >=1.5 or <=-1.5
            # filtered_genes_by_score = filtered_genes[(filtered_genes['Score'] >= 1.5) | (filtered_genes['Score'] <= -1.5)]
            # Filter genes by P-value threshold and calculate mean score
            filtered_genes_by_p_value = filtered_genes[filtered_genes['P-value'] <= 0.05]
            if not filtered_genes_by_p_value.empty:
                pathway_mean_scores[pathway] = filtered_genes_by_p_value['Score'].mean()
            else:
                pathway_mean_scores[pathway] = 0

    return enriched_pathway_dict, enriched_pathway_genes, pathway_mean_scores


def calculate_trend(pathway_mean_scores):
    """Calculates the trend (up or down) of each pathway based on its mean score."""
    trends = {}
    for pathway, mean_scores in pathway_mean_scores.items():
        # Calculate the overall mean score for the pathway across all conditions
        overall_mean_score = np.mean(mean_scores)
        trend = "Up" if overall_mean_score > 0 else "Down"
        trends[pathway] = trend
    return trends



def bold_keywords(text, keywords):
    """Returns text with keywords in bold."""
    for keyword in keywords:
        if keyword in text:
            text = text.replace(text, f"**{text}**")
    return text


def print_aggregated_pathway_information(aggregated_data, output_dir, experiment_name):
    file_path = path.join(output_dir, 'Text', f'{experiment_name}_aggregated.txt')
    with open(file_path, 'w') as file:
        for pathway, details in aggregated_data.items():
            file.write(f"Pathway: {pathway}\n")
            file.write(f"Trends: {', '.join(details['trends'])}\n")
            file.write("Significant Genes:\n")
            for gene_id, gene_info in details['significant_genes'].items():
                file.write(f"    Symbol: {gene_info['Symbol']}, Score: {gene_info['Score']}\n")
            file.write("\n")



def plot_pathways_mean_scores(all_mean_scores, output_dir, experiment_name):
    """Plots mean scores of pathways across experiments in a grouped bar chart."""

    data_df = pd.DataFrame(all_mean_scores)

    # Increase the figure size significantly
    plt.figure(figsize=(60, 20))  # Adjust the size as needed
    ax = plt.subplot(111)

    conditions = list(all_mean_scores.keys())
    total_pathways = data_df.index
    num_conditions = len(conditions)
    bar_width = 0.8 / num_conditions
    positions = np.arange(len(total_pathways))

    for i, condition in enumerate(conditions):
        mean_scores = data_df[condition].values
        ax.bar(positions + bar_width * i, mean_scores, width=bar_width, label=condition)

    ax.set_xticks(positions + bar_width * (num_conditions / 2) - bar_width / 2)
    ax.set_xticklabels(total_pathways, rotation=90, fontsize=14)
    ax.set_xlabel('Pathways', fontsize=16)
    ax.set_ylabel('Mean Scores', fontsize=16)
    ax.set_title('Pathway Mean Scores Across Different Conditions', fontsize=20)
    ax.legend(prop={'size': 14})

    plt.subplots_adjust(bottom=0.4)  # Adjust for layout

    output_file_path = path.join(output_dir, f"{experiment_name}_pathway_scores.pdf")
    file_path = path.join(output_dir, 'Plots', f'{experiment_name}_plot.txt')
    plt.savefig(output_file_path, format='pdf', bbox_inches='tight')
    plt.show()



genes_by_pathway = load_pathways_genes()
test_list = ['T_v_N', '500nm_v_T']
for test_name in test_list:
    print(f"running enrichment on {test_name}")
    significant_pathways_with_genes, scores = perform_statist(test_name)
    filtered_pathways = perform_statist_mann_whitney(significant_pathways_with_genes, scores)
    print_enriched_pathways_to_file(filtered_pathways, temp_output_folder, FDR_THRESHOLD)

print("finished enrichment")
test_file_paths = [f'{input_dir}/T_v_N.csv', f'{input_dir}/500nm_v_T.csv']
import os
# Extract the test names from the file paths to match them with condition files
test_names = [os.path.splitext(os.path.basename(path))[0] for path in test_file_paths]

# Get the list of condition files
condition_files_unsorted = [os.path.join(temp_output_folder, file) for file in os.listdir(temp_output_folder)]

# Sort condition_files based on the order of test_names
condition_files = sorted(condition_files_unsorted, key=lambda x: test_names.index(os.path.splitext(os.path.basename(x))[0]))

# Initialize dictionaries to store aggregated data
all_enriched_genes = {}
all_mean_scores = {}
aggregated_pathway_data = {}

# Processing and aggregating data
for condition_file, experiment_file in zip(condition_files, test_file_paths):
    scores_dict, pathway_genes_dict, pathway_mean_scores = process_experiment(condition_file, experiment_file, pathway_file_dir)
    pathway_trends = calculate_trend(pathway_mean_scores)

    for pathway, p_value in scores_dict.items():
        if pathway not in aggregated_pathway_data:
            aggregated_pathway_data[pathway] = {'trends': [], 'significant_genes': {}}

        condition_name = path.basename(condition_file).split('.')[0]
        aggregated_pathway_data[pathway]['trends'].append(f"{condition_name}: {pathway_trends.get(pathway, 'N/A')}")

        # Aggregate significant genes across conditions
        for gene_id, gene_details in pathway_genes_dict.get(pathway, {}).items():
            if gene_details['P-value'] <= P_VALUE_THRESHOLD:  # Check if the gene is significant
                if gene_id not in aggregated_pathway_data[pathway]['significant_genes']:
                    aggregated_pathway_data[pathway]['significant_genes'][gene_id] = gene_details
    all_enriched_genes[condition_file] = scores_dict
    all_mean_scores[condition_file] = pathway_mean_scores

# Print aggregated pathway information
print_aggregated_pathway_information(aggregated_pathway_data, output_path, Experiment_name)

# Plot the mean scores
plot_pathways_mean_scores(all_mean_scores, output_path, Experiment_name)

# Clean up the output folder if it exists
if path.exists(temp_output_folder):
    shutil.rmtree(temp_output_folder)


