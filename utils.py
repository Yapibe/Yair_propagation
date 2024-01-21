import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
import os
from os import path
import pickle
import zlib
import args
import openpyxl

def save_filtered_pathways_to_tsv(pathways_with_many_genes, genes_by_pathway, output_file_path):
    """
    Saves the filtered pathways with their gene counts and gene IDs to a TSV file.

    Args:
        pathways_with_many_genes (list): List of pathway names that passed filtering.
        genes_by_pathway (dict): Original dictionary mapping pathways to their gene IDs.
        output_file_path (str): Path to save the TSV file.

    Returns:
        None
    """
    with open(output_file_path, 'w') as file:
        for pathway in pathways_with_many_genes:
            genes = genes_by_pathway.get(pathway, [])
            gene_count = len(genes)
            line = f"{pathway}\t{gene_count}\t{' '.join(map(str, genes))}\n"
            file.write(line)
    print(f"Filtered pathways saved to {output_file_path}")


def filter_network_by_prior_data(network_filename, prior_data):
    """
    Filters a network to only include nodes present in the prior_data DataFrame.

    Args:
        network_filename (str): Path to the network file.
        prior_data (pd.DataFrame): DataFrame containing gene information.

    Returns:
        networkx.Graph: A filtered graph object.
    """
    # Read the network
    network = read_network(network_filename)

    # Get the set of gene IDs from prior_data
    gene_ids_in_prior_data = set(prior_data['GeneID'])

    # Filter the network
    filtered_network = network.copy()
    for node in network.nodes:
        if node not in gene_ids_in_prior_data:
            filtered_network.remove_node(node)

    return filtered_network


# noinspection PyTypeChecker
def read_network(network_filename):
    """
    Reads a network from a file and returns a NetworkX graph.

    Args:
        network_filename (str): Path to the file containing the network data.

    Returns:
        networkx.Graph: A graph object representing the network.
    """
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2])
    return nx.from_pandas_edgelist(network, 0, 1, 2)


def read_prior_set(excel_dir):
    """
    Reads prior data set from an Excel file and applies preprocessing.
    Args:
        excel_dir (str): Path to the Excel file containing the prior data.
    Returns:
        pandas.DataFrame: DataFrame containing the preprocessed prior data.
    """
    prior_data = pd.read_excel(excel_dir, engine='openpyxl')

    # Drop duplicate GeneID values
    # print all duplicates
    print(prior_data[prior_data.duplicated(subset=['GeneID'])])
    prior_data = prior_data.drop_duplicates(subset='GeneID')
    # remove any row with no value in Score column
    prior_data = prior_data[prior_data['Score'].notna()]
    # remove any row with "?" in Score column
    prior_data = prior_data[prior_data['Score'] != '?']
    # Filter out GeneIDs that are not purely numeric (to exclude concatenated IDs)
    prior_data = prior_data[prior_data['GeneID'].apply(lambda x: str(x).isdigit())]
    # Convert GeneID to integer
    prior_data['GeneID'] = prior_data['GeneID'].astype(int)

    # Reset the DataFrame index
    prior_data = prior_data.reset_index(drop=True)

    return prior_data


def get_propagation_input(prior_gene_ids, prior_data, input_type):
    """
    Generates propagation inputs based on specified type and network.

    Args:
        prior_gene_ids (set): List of gene IDs for the prior set.
        prior_data (pandas.DataFrame): DataFrame containing all experimental data.
        input_type (str): Type of input to generate (e.g., 'ones', 'abs_Score', etc.).

    Returns:
        dict: A dictionary mapping gene IDs to their corresponding input values.
    """
    inputs = dict()
    if input_type == 'ones' or input_type is None:
        inputs = {int(x): 1 for x in prior_gene_ids}
    elif input_type == 'abs_Score':
        for id in prior_gene_ids:
            try:
                # Your original code for processing each id
                inputs[int(id)] = np.abs(float(prior_data[prior_data.GeneID == id]['Score'].values[0]))
            except TypeError:
                print(f"Error processing ID: {id}")
                print(f"Data for this ID: {prior_data[prior_data.GeneID == id]}")
                # Optionally, you can break or continue based on your needs
                break

    elif input_type == 'Score':
        inputs = {int(id): float(prior_data[prior_data.GeneID == id]['Score'].values[0]) for id in prior_gene_ids}
    else:
        assert 0, '{} is not a valid input type'.format(input_type)

    inputs = {id: np.round(input_score, 3) for id, input_score in inputs.items()}
    return inputs


def save_file(obj, save_dir=None, compress=True):
    """
    Saves an object to a file, with optional compression.
    Args:
        obj (object): The object to be saved.
        save_dir (str, optional): The directory where the file will be saved.
        compress (bool, optional): Whether to compress the file.
    Returns:
        None
    """
    obj = pickle.dumps(obj)
    if compress:
        obj = zlib.compress(obj)
    with open(save_dir, 'wb') as f:
        pickle.dump(obj, f)
    print('File was saved in {}'.format(save_dir))


def save_propagation_score(propagation_scores, prior_set, propagation_input, genes_id_to_idx, task, save_dir=None):
    """
    Saves the propagation scores to a file.

    Args:
        propagation_scores (ndarray): The propagation scores.
        prior_set (dataframe): The set of prior genes.
        propagation_input (dict): The input used for propagation.
        genes_id_to_idx (dict): Mapping from gene indices to gene IDs.
        task (PropagationTask): The propagation task object containing program arguments.
        save_dir (str, optional): Directory to save the results.

    Returns:
        dict: A dictionary containing the saved data.
    """
    file_name = f"{task.test_name}_{task.propagation_input_type}_{task.alpha}_{task.date}"
    save_dir = save_dir or task.propagation_scores_path
    os.makedirs(save_dir, exist_ok=True)
    propagation_results_path = path.join(save_dir, file_name)

    save_dict = {
        'args': task, 'prior_set': prior_set, 'propagation_input': propagation_input,
        'gene_id_to_idx': genes_id_to_idx, 'gene_prop_scores': propagation_scores,
    }

    save_file(save_dict, propagation_results_path)
    return save_dict


def get_root_path():
    """
    Retrieves the root path of the current script.
    Returns:
        str: The root directory path of the current script file.
    """
    return path.dirname(path.realpath(__file__))


def load_pathways_genes(pathways_dir):
    """
    Loads the pathways and their associated genes from a file.

    Args:
        pathways_dir (str): Path to the file containing the pathway data.

    Returns:
        dict: A dictionary mapping pathway names to lists of genes in each pathway.
    """
    # with open(pathways_dir, 'r') as f:
    #     lines = [str.upper(x.strip()).split('\t') for x in f]
    # pathways = {x[0]: [int(y) for y in x[2:]] for x in lines}
    #
    # return pathways

    pathways = {}
    with open(pathways_dir, 'r') as f:
        for line in f:
            parts = line.strip().upper().split('\t')  # Split each line into parts
            if len(parts) < 3:  # If there are not enough parts for name, size, and genes
                continue

            pathway_name = parts[0]  # The pathway name is the first part
            try:
                pathway_size = int(parts[1])  # The pathway size is the second part
            except ValueError:
                continue  # Skip this line if the size is not an integer

            # Further split the gene part by spaces and then take the number of genes specified by pathway_size
            genes = parts[2]  # The third part is the space-separated list of gene IDs
            gene_list = genes.split()  # Split the genes by spaces

            # Convert each gene to an integer
            try:
                genes = [int(gene) for gene in gene_list[:pathway_size]]
            except ValueError:
                continue  # Skip this line if any gene is not an integer

            pathways[pathway_name] = genes  # Add the pathway and its genes to the dictionary

    return pathways


def load_file(load_dir, decompress=True):
    """
    Loads an object from a file using pickle with optional decompression.

    Args:
        load_dir (str): The directory from which to load the file.
        decompress (bool, optional): Whether to decompress the file.

    Returns:
        file (dict): The object loaded from the file.
    """
    with open(load_dir, 'rb') as f:
        file = pickle.load(f)
    if decompress:
        try:
            file = pickle.loads(zlib.decompress(file))
        except zlib.error:
            print('Entered an uncompressed file but asked to decompress it')
    return file


def load_propagation_scores(task, propagation_file_name=None):
    """
    Loads the propagation scores from a file.

    Args:
        task (PropagationTask): The propagation task object containing program arguments.
        propagation_file_name (str, optional): Name of the file to load the results from.

    Returns:
        dict: Dictionary containing the propagation scores, input, and gene index to ID mapping.
    """
    propagation_results_path = path.join(task.propagation_scores_path, propagation_file_name)
    propagation_results_path = propagation_results_path.replace("\\", "/")  # Replace backslashes

    propagation_result_dict: dict = load_file(propagation_results_path, decompress=True)

    return propagation_result_dict


def shuffle_scores(dataframe, shuffle_column, num_shuffles=10):
    if shuffle_column not in dataframe.columns:
        raise ValueError(f"Column '{shuffle_column}' not found in the DataFrame")

    # Dictionary to store all shuffled columns
    shuffled_columns = {}

    # Generate shuffled columns with progress tracking
    for i in tqdm(range(1, num_shuffles + 1), desc='Shuffling Scores'):
        shuffled_column = dataframe[shuffle_column].sample(frac=1).reset_index(drop=True)
        shuffled_columns[f'Shuffled_Score_{i}'] = shuffled_column

    # Convert the dictionary to a DataFrame
    shuffled_df = pd.DataFrame(shuffled_columns)

    # Concatenate the original dataframe with the new shuffled scores DataFrame
    result_df = pd.concat([dataframe, shuffled_df], axis=1)

    return result_df


def load_network_and_pathways(task):
    """
    Loads the network graph and pathways based on the provided configuration.
    Returns:
        tuple: A tuple containing the network graph, a list of interesting pathways, and a dictionary mapping
               pathways to their genes.
    """
    # network_graph = read_network(general_args.network_file_path)
    genes_by_pathway = load_pathways_genes(task.pathway_file_dir)
    scores = get_scores(task)

    return genes_by_pathway, scores


def get_scores(task):
    # Path to the file containing the raw scores (adjust as necessary)
    raw_scores_file_path = task.experiment_file_path

    try:
        # Load raw data from the file
        raw_data = pd.read_excel(raw_scores_file_path)

        # Perform necessary preprocessing on raw_data
        # For instance, sorting, filtering, or extracting specific columns
        # Assuming 'GeneID' and 'Score' are columns in the raw data
        sorted_raw_data = raw_data.sort_values(by='GeneID').reset_index(drop=True)

        # Create a dictionary for gene_id_to_score
        scores_dict = {gene_id: score for gene_id, score in zip(sorted_raw_data['GeneID'], sorted_raw_data['Score'])}
        return scores_dict

    except FileNotFoundError:
        print(f"File not found: {raw_scores_file_path}")
        return pd.DataFrame(), {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), {}

def load_and_prepare_data(task):
    data = read_prior_set(task.experiment_file_path)
    # abs score
    data['Score'] = data['Score'].apply(lambda x: abs(x))
    data = data[data['GeneID'].apply(lambda x: str(x).isdigit())]
    data['GeneID'] = data['GeneID'].astype(int)
    # remove p_values column
    data = data.drop(columns=['P-value'])
    data = data.reset_index(drop=True)

    return data
