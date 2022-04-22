"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for extracting binding core examples and writing their corresponding distance matrix/ sequence encodings.
"""

#imports
from prody import *
import numpy as np
import os
import pickle
import itertools

def get_neighbors(structure, coordinating_resind: int, no_neighbors: int):
    """Finds neighbors of an input coordinating residue.

    Args:
        coordinating_resnum (int): Residue number of coordinatign residue.
        start_resnum (int): Very first residue number in input structure.
        end_resnum (int): Very last residue number in input structure.

    Returns:
        core_fragment (list): List containing resnumbers of coordinating residue and neighbors. 
    """

    chain_id = list(set(structure.select(f'resindex {coordinating_resind}').getChids()))[0]
    all_resinds = structure.select(f'chain {chain_id}').select('protein').getResindices()
    terminal = max(all_resinds)
    start = min(all_resinds)

    extend = np.array(range(-no_neighbors, no_neighbors+1))
    _core_fragment = np.full((1,len(extend)), coordinating_resind) + extend
    core_fragment = [ind for ind in list(_core_fragment[ (_core_fragment >= start) & (_core_fragment <= terminal) ]) if ind in all_resinds] #remove nonexisting neighbor residues

    return core_fragment

def generate_filename(parent_structure_id: str, binding_core_resis: list, metal: tuple):
    """Helper function for generating file names.

    Args:
        parent_structure_id (str): The pdb identifier of the parent structure.
        binding_core_resis (list): List of residue numbers that comprise the binding core.
        filetype (str): The type of file.
        extension (str): The file extension.
        metal (tuple): A tuple containing the element symbol of the metal in all caps and the residue number of said metal. 
    """

    filename = parent_structure_id + '_' + '_'.join([str(num) for num in binding_core_resis]) + '_' + metal[0] + str(metal[1])
    return filename

def write_pdb(core, out_dir: str, filename: str):
    """Generates a pdb file for an input core

    Args:
        core (prody.atomic.atomgroup.AtomGroup): AtomGroup of binding core.
        metal (str): The element symbol of the bound metal in all caps.
    """

    writePDB(os.path.join(out_dir, filename + '_core.pdb'), core) #write core to a pdb file

def get_contiguous_resnums(resnums: np.ndarray):
    resnums = list(resnums)
    temp = resnums[:]
    fragments = []
    for resnum in temp:
        fragment = []
        temp.remove(resnum)
        fragment.append(resnum)
        queue = [i for i in temp if abs(i-resnum)==1]

        while len(queue) != 0:
            current = queue.pop()
            fragment.append(current)
            temp.remove(current)
            queue += [i for i in temp if abs(i-current)==1]

        fragment.sort()
        fragments.append(fragment)

    fragment_indices = []
    for fragment in fragments:
        fragment_indices.append([resnums.index(i) for i in fragment])
    
    return fragment_indices

def permute_features(dist_mat: np.ndarray, encoding: np.ndarray, label: np.ndarray, resnums: np.ndarray):
    all_features = {}
    full_observations = []
    full_labels = []

    fragment_indices = get_contiguous_resnums(resnums)
    fragment_index_permutations = itertools.permutations(list(range(0,len(fragment_indices))))
    atom_indices = np.split(np.linspace(0, len(resnums)*4-1, len(resnums)*4), len(resnums))
    for index, index_permutation in enumerate(fragment_index_permutations):
        feature = {}
        permutation = sum([fragment_indices[i] for i in index_permutation], [])
        atom_ind_permutation = sum([list(atom_indices[i]) for i in permutation], [])
        permuted_dist_mat = np.zeros(dist_mat.shape)

        for i, atom_indi in enumerate(atom_ind_permutation):
            for j, atom_indj in enumerate(atom_ind_permutation):
                permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]

        if index == 0:
            for i in range(0,permuted_dist_mat.shape[0]):
                for j in range(0, permuted_dist_mat.shape[1]):
                    assert permuted_dist_mat[i,j] == dist_mat[i,j]

            
        split_encoding = np.array_split(encoding.squeeze(), encoding.shape[1]/20)
        _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in index_permutation], []), [])
        zeros = np.zeros(20 * (len(split_encoding) - len(resnums)))
        permuted_encoding = np.concatenate((_permuted_encoding, zeros))

        permuted_label = []
        for i in index_permutation:
            frag = fragment_indices[i]
            for j in range(0,len(frag)):
                ind = frag[j]
                permuted_label += list(label[4*ind:4*ind+4])
        permuted_label = np.array(permuted_label)

        feature['distance'] = permuted_dist_mat
        feature['encoding'] = permuted_encoding
        feature['label'] = permuted_label

        full_observations.append(list(np.concatenate((permuted_dist_mat.flatten(), permuted_encoding))))
        full_labels.append(list(permuted_label))

        all_features[index] = feature
    all_features['full_observations'] = np.array(full_observations)
    all_features['full_labels'] = np.array(full_labels)
    return all_features

def write_features(features: dict, out_dir: str, filename: str):
    """Writes a pickle file to hold feature information for a given core.

    Args:
        features (dict): Holds distance matrices for a given core.
    """

    with open(os.path.join(out_dir, filename + '_features.pkl'), 'wb') as f:
        pickle.dump(features, f)

def extract_cores(pdb_file: str, no_neighbors: int, coordinating_resis: int):
    """Finds all putative metal binding cores in an input protein structure.

    Returns:
        cores (list): List of metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
        metal_names (list): List of metal names indexed by binding core.
    """

    metal_sel = f'name NI MN ZN CO CU MG FE'
    structure = parsePDB(pdb_file) #load structure

    cores = []
    names = []

    metal_resindices = structure.select('hetero').select(metal_sel).getResindices()
    metal_names = structure.select('hetero').select(metal_sel).getNames()

    for metal_ind, name in zip(metal_resindices, metal_names):

        try: #try/except to account for solvating metal ions included for structure determination
            coordinating_resindices = list(set(structure.select(f'protein and not carbon and not hydrogen and within 2.83 of resindex {metal_ind}').getResindices()))

        except:
            continue
        
        if len(coordinating_resindices) <= coordinating_resis and len(coordinating_resindices) >= 2:
            binding_core_resindices = []
            for ind in coordinating_resindices:
                core_fragment = get_neighbors(structure, ind, no_neighbors)
                binding_core_resindices += core_fragment

            binding_core_resindices.append(metal_ind)
            binding_core = structure.select('resindex ' + ' '.join([str(num) for num in binding_core_resindices]))
            cores.append(binding_core)
            names.append(name)

        else:
            continue
    return cores, names

def remove_degenerate_cores(cores: list, metal_names: list):
    """Function to remove cores that are the same. For example, if the input structure is a homotetramer, this function will only return one of the binding cores.

    Args:
        cores (list): List of metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.

    Returns:
        unique_cores (list): List of all unique metal binding cores. Each element is a prody.atomic.atomgroup.AtomGroup object.
        unique_names (list): List of all metal names indexed by unique binding core.
    """

    #TODO Update method to do structural alignment. For some reason, ProDy was not doing this properly.

    if len(cores) > 1:
        unique_cores = []
        unique_names = []
        while cores:
            current_core = cores.pop() #extract last element in cores
            current_name = metal_names.pop()
            current_total_atoms = len(current_core.getResnums())
            current_core.setChids('A')


            pairwise_seqids = np.array([])
            pairwise_overlap = np.array([])

            for core in cores: #iterate through all cores 
                core.setChids('B')
                if current_total_atoms == len(core.getResnums()): #if the current cores and core have the same number of atoms, compute RMSD    
                    reference, target, seqid, overlap = matchChains(current_core.select('protein'), core.select('protein'))[0]
                    pairwise_seqids = np.append(pairwise_seqids, seqid)
                    pairwise_overlap = np.append(pairwise_overlap, overlap)

                else:
                    continue

            degenerate_core_indices = list(set(np.where(pairwise_seqids == 100)[0]).intersection(set(np.where(pairwise_overlap == 100)[0]))) #find all cores that are essentially the same structure


            if len(degenerate_core_indices) > 0: #remove all degenerate cores from cores list
                for ind in degenerate_core_indices:
                    del cores[ind]
                    del metal_names[ind]

            unique_cores.append(current_core) #add reference core 
            unique_names.append(current_name)

    else:
        unique_cores = cores 
        unique_names = metal_names

    return unique_cores, unique_names

def compute_labels(core, metal_name: str, no_neighbors, coordinating_resis):
    """Given a metal binding core, will compute the distance of all backbone atoms to metal site.

    Returns:
        distances (np.ndarray): A 1xn array containing backbone distances to metal, where n is the number of residues in the binding core. As an example, elements 0:4 contain distances between the metal and CA, C, O, and CB atoms of the binding core residue of lowest resnum.
    """

    metal_sel = core.select('hetero').select(f'name {metal_name}')
    binding_core_backbone = core.select('protein').select('name CA C O N')
    distances = buildDistMatrix(metal_sel, binding_core_backbone)

    max_atoms = 4 * (coordinating_resis + (2*coordinating_resis*no_neighbors)) #standardize shape of label matrix
    padding = max_atoms - distances.shape[1]
    distances = np.lib.pad(distances, ((0,0),(0,padding)), 'constant', constant_values=0)
    return distances 

def compute_distance_matrices(core, no_neighbors: int, coordinating_resis: int):
    """Generates binding core backbone distances and label files.

    Returns:
        distance_matrices (dict): Dictionary containing distance matrices and a numpy array of resnums that index these matrices.
    """

    distance_matrix = {}
    binding_core_resnums = core.select('protein').select('name N').getResnums()

    max_atoms = 4 * coordinating_resis + (2*coordinating_resis*no_neighbors)
    binding_core_backbone = core.select('protein').select('name CA O C N')
    full_dist_mat = buildDistMatrix(binding_core_backbone, binding_core_backbone)
    
    padding = max_atoms - full_dist_mat.shape[0]
    full_dist_mat = np.lib.pad(full_dist_mat, ((0,padding), (0,padding)), 'constant', constant_values=0)

    distance_matrix['dist'] = full_dist_mat
    distance_matrix['resnums'] = binding_core_resnums

    return distance_matrix

def onehotencode(core, no_neighbors: int, coordinating_resis: int):
    """Adapted from Ben Orr's function from make_bb_info_mats, get_seq_mat. Generates one-hot encodings for sequences.

    Returns:
        seq_channels (np.ndarray): array of encodings for 
    """
    seq = core.select('name CA').getResnames()
    threelettercodes = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', \
                        'MET','PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

    encoding = np.array([[]])

    for i in range(len(seq)):
        aa = str(seq[i])
        try:
            idx = threelettercodes.index(aa)
            one_hot = np.zeros((1,20))
            one_hot[0,idx] = 1
        except:
            print('Resname of following atom not found: {}'.format(aa))
            continue

        encoding = np.concatenate((encoding, one_hot), axis=1)

    max_resis = coordinating_resis +  (coordinating_resis * no_neighbors * 2)
    padding = 20 * (max_resis - len(seq))
    encoding = np.concatenate((encoding, np.zeros((1,padding))), axis=1)

    return encoding
    
def construct_training_example(pdb_file: str, output_dir: str, no_neighbors=1, coordinating_resis=4):
    """For a given pdb file, constructs a training example and extracts all features.

    Args:
        pdb_file (str): Path to input pdb file.
        output_dir (str): Path to output directory.
        no_neighbors (int, optional): Number of neighbors in primary sequence to coordinating residues be included in core. Defaults to 1.
        coordinating_resis (int, optional): Sets a threshold for maximum number of metal coordinating residues. Defaults to 4.
    """

    cores, names = extract_cores(pdb_file, no_neighbors, coordinating_resis)
    unique_cores, unique_names = remove_degenerate_cores(cores, names)

    for core, name in zip(unique_cores, unique_names):
        label = compute_labels(core, name, no_neighbors, coordinating_resis)
        distance_matrix = compute_distance_matrices(core, no_neighbors, coordinating_resis)
        encoding = onehotencode(core, no_neighbors, coordinating_resis)

        features = permute_features(distance_matrix, encoding, label)

        metal_resnum = core.select(f'name {name}').getResnums()[0]
        filename = generate_filename(core.getTitle(), core.select('name N').getResnums(), (name, metal_resnum))

        write_pdb(core, output_dir, filename)
        write_features(features, output_dir, filename)