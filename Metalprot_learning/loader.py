"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading metal binding cores from input PDB files of positive examples.

In brief, the extract_cores function takes in a PDB file and outputs all metal binding cores. In this context, we define a metal binding core as the coordinating residues and 
their neighbors in primary sequence. This function calls get_neighbors to identify neighbors in primary sequence. remove_degenerate_cores, as implied by the name, removes cores
found in extract_cores that are the same. For example, homomeric metalloproteins (i.e. hemoglobin) may have mutiple equivalent binding sites, which would be removed via action
of remove_degenerate_cores.
"""

import os
from turtle import distance
import numpy as np
import pandas as pd
import itertools 
from scipy.spatial.distance import cdist
from pypivoter.degeneracy_cliques import enumerateCliques
from Metalprot_learning.utils import AlignmentError, EncodingError, PermutationError
from prody import parsePDB, AtomGroup, matchChains, buildDistMatrix, writePDB

def remove_degenerate_cores(cores: list):
    """
    Function to remove cores that are the same. For example, if the input 
    structure is a homotetramer, this function will only return one of the binding cores.
    """
    try:
        if len(cores) > 1:
            unique_cores = []
            while cores:
                ref = cores.pop() #extract last element in cores
                ref_total_atoms = ref.structure.select('protein').numAtoms()
                ref_resis = set(ref.structure.select('protein').select('name CA').getResnames())
                ref_length = len(ref_resis)

                pairwise_seqids, pairwise_overlap = np.array([]), np.array([])
                for core in cores: #iterate through all cores 
                    total_atoms = core.structure.select('protein').numAtoms()
                    resis = set(core.structure.select('protein').select('name CA').getResnames())
                    length = len(resis)

                    #if the reference and core have the same number of atoms, quantify similarity
                    if ref_total_atoms == total_atoms and ref_resis == resis and ref_length == length:    
                        try:
                            _, _, seqid, overlap = matchChains(ref.structure.select('protein'), core.structure.select('protein'))[0]
                            pairwise_seqids, pairwise_overlap = np.append(pairwise_seqids, seqid), np.append(pairwise_overlap, overlap)

                        except:
                            pairwise_seqids, pairwise_overlap = np.append(pairwise_seqids, 0), np.append(pairwise_overlap, 0)

                    else:
                        pairwise_seqids, pairwise_overlap = np.append(pairwise_seqids, 0), np.append(pairwise_overlap, 0)

                degenerate_core_indices = list(set(np.where(pairwise_seqids == 100)[0]).intersection(set(np.where(pairwise_overlap == 100)[0]))) #find all cores that are essentially the same structure

                if len(degenerate_core_indices) > 0: #remove all degenerate cores from cores list
                    cores = [cores[i] for i in range(0,len(cores)) if i not in degenerate_core_indices]

                unique_cores.append(ref) #add reference core 

        else:
            unique_cores = cores

    except:
        raise AlignmentError

    return unique_cores

def _impute_ca_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray):
    """
    Helper function for imputing CB coordinates. Returns an imputed set of CB coordinates.
    """
    B = ca - n
    C = c - ca
    A = np.cross(B, C, axis = 1)
    return (-0.58273431 * A) + (0.56802827 * B) - (0.54067466 * C)

class Core:
    def __init__(self, source: str, core: AtomGroup, coordinating_resis: np.ndarray, identifiers: list, sequence: np.ndarray, putative: bool):
        self.structure, self.identifiers, self.source, self.sequence, self.putative, self.coordinating_resis = core, identifiers, source, sequence, putative, coordinating_resis
        self.label = self._define_label()
        self.permuted_labels, self.permuted_identifiers, self.permuted_coordination_labels = [None] * 3
        self.coordination_label = self._define_coordination_label()
        self.metal_coords, self.metal_name = self._define_metal()
        self.filename = self._define_filename()

    def _define_label(self, distogram=False):
        if self.putative:
            label = None

        elif not distogram:
            label = np.zeros(12*4)
            _label = buildDistMatrix(self.structure.select('protein').select('name N CA C O'), self.structure.select('hetero')).squeeze()
            label[0:len(_label)] = _label

        else:
            distances = buildDistMatrix(self.structure.select('protein').select('name N CA C O'), self.structure.select('hetero')).squeeze()
            bins = np.arange(0, 12.5, 0.1)
            label = np.zeros(len(48), len(bins))
            for ind, distance in enumerate(distances):
                label[ind] = np.histogram(distance, bins)[0]
        return label

    def _define_coordination_label(self):
        coordination_label = None if self.putative else np.array([1 if i in self.coordinating_resis else 0 for i in self.structure.select('name N').getResindices()]) 
        return coordination_label

    def _define_metal(self):
        if self.putative:
            metal_coords, metal_name = np.array([np.nan] * 3), ''
        else:
            metal = self.structure.select('hetero')
            metal_coords = metal.getCoords()[0]
            metal_name = metal.getResnames()[0] + str(metal.getResnums()[0]) + metal.getChids()[0]
        return metal_coords, metal_name

    def _define_filename(self):
        prefix = 'PUTATIVE_' if self.putative else ''
        suffix = '' if self.putative else '_' + self.metal_name
        filename = prefix + self.structure.getTitle() + '_' + '_'.join([str(tup[0]) + tup[1] for tup in self.identifiers]) + suffix
        return filename

    def _identify_fragments(self):
        binding_core_identifiers = self.identifiers
        temp = binding_core_identifiers[:]
        fragments = []
        while len(temp) != 0:
            for i in range(0, len(temp)): #build up contiguous fragments by looking for adjacent resnums
                if i == 0:
                    fragment = [temp[i]]

                elif set(temp[i][1]) == set([i[1] for i in fragment]) and 1 in set([abs(temp[i][0] - j[0]) for j in fragment]):
                    fragment.append(temp[i])

            fragment = list(set(fragment)) 
            fragment.sort()
            fragment_indices = [binding_core_identifiers.index(i) for i in fragment] 
            fragments.append(fragment_indices) #build a list containing lists of indices of residues for a given fragment

            for item in fragment:
                temp.remove(item)
    
        return fragments

    def _permute_labels(self, permutation, fragment_indices, atom_indices):
        permuted_label = np.zeros(len(self.label))
        _permuted_label = np.array([])
        for i in permutation:
            frag = fragment_indices[i]
            for j in frag:
                atoms = atom_indices[j]
                for atom in atoms:
                    _permuted_label = np.append(_permuted_label, self.label[int(atom)])
        permuted_label[0:len(_permuted_label)] = _permuted_label
        return permuted_label

    def write_pdb_files(self, output_dir: str):
        writePDB(os.path.join(output_dir, self.filename + '.pdb.gz'), self.structure)

class FCNCore(Core):
    def __init__(self, source: str, core: AtomGroup, coordinating_resis: np.ndarray, identifiers: list, sequence, distance_matrix: np.ndarray, putative: bool):
        super().__init__(source, core, coordinating_resis, identifiers, sequence, putative)
        self.distance_matrix, self.encoding = distance_matrix, FCNCore._onehotencode(self.sequence)
        self.permuted_distance_matrices, self.permuted_encodings = [None], [None]

    @staticmethod
    def _onehotencode(sequence: np.ndarray):
        threelettercodes = {'ALA': 0 , 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'CSO': 4,'GLU': 5, 'GLX': 5,'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10,
                            'LYS': 11, 'MET': 12, 'MSE': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'SEP': 15, 'THR': 16, 'TPO': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
        encoding = np.array([[]])
        for i in range(len(sequence)):
            aa = str(sequence[i])
            if aa not in threelettercodes:
                print(aa)
                raise EncodingError
            idx = threelettercodes[aa]
            one_hot = np.zeros((1,20))
            one_hot[0,idx] = 1
            encoding = np.concatenate((encoding, one_hot), axis=1)
        max_resis = 12
        padding = 20 * (max_resis - len(sequence))
        return np.concatenate((encoding, np.zeros((1,padding))), axis=1).squeeze()

    @staticmethod
    def _permute_matrices(dist_mat: np.ndarray, atom_ind_permutation):
        permuted_dist_mat = np.zeros(dist_mat.shape)
        for i, atom_indi in enumerate(atom_ind_permutation):
            for j, atom_indj in enumerate(atom_ind_permutation):
                permuted_dist_mat[i,j] = dist_mat[int(atom_indi), int(atom_indj)]
        return permuted_dist_mat

    @staticmethod
    def _permute_encodings(encoding: np.ndarray, fragment_indices, permutation):
        permuted_encoding = np.zeros(len(encoding))
        split_encoding = np.array_split(encoding.squeeze(), len(encoding)/20)
        _permuted_encoding = sum(sum([[list(split_encoding[i]) for i in fragment_indices[j]] for j in permutation], []), []) #permute the encoding by fragment
        permuted_encoding[0:len(_permuted_encoding)] = _permuted_encoding
        return permuted_encoding

    def permute(self, trim=False):
        permuted_dist_mats, permuted_encodings, permuted_labels, permuted_identifiers, permuted_coordinating_resis = [], [], [], [], []
        fragment_indices = self._identify_fragments()
        fragment_permutations = itertools.permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
        no_atoms = len(self.structure.select('protein').select('name N CA C O'))
        atom_indices = np.split(np.linspace(0, no_atoms - 1, no_atoms), no_atoms / 4)
        for permutation in fragment_permutations:
            fragment_index_permutation = sum([fragment_indices[i] for i in permutation], [])
            atom_index_permutation = sum([list(atom_indices[i]) for i in fragment_index_permutation], []) 
            _distance_matrix_permutation = FCNCore._permute_matrices(self.distance_matrix, atom_index_permutation)
            distance_matrix_permutation = _distance_matrix_permutation.flatten() if not trim else _distance_matrix_permutation[np.triu_indices(_distance_matrix_permutation.shape[0], k=1)].flatten()
            encoding_permutation = FCNCore._permute_encodings(self.encoding, fragment_indices, permutation)

            permuted_dist_mats.append(distance_matrix_permutation)
            permuted_encodings.append(encoding_permutation)
            permuted_labels.append(self._permute_labels(permutation, fragment_indices, atom_indices))
            permuted_identifiers.append([self.identifiers[i] for i in fragment_index_permutation])
            permuted_coordinating_resis.append([self.coordination_label[i] for i in fragment_index_permutation])

        self.permuted_distance_matrices = permuted_dist_mats
        self.permuted_encodings =permuted_encodings
        self.permuted_labels = permuted_labels
        self.permuted_identifiers = permuted_identifiers
        self.permuted_coordination_labels = permuted_coordinating_resis

    def write_data_files(self, output_dir: str):
        suffix = '_FCN_features.pkl'
        filename = self.filename + suffix
        if self.permuted_distance_matrices and self.permuted_labels and self.permuted_identifiers and self.permuted_coordination_labels:
            df = pd.DataFrame({'distance_matrices': self.permuted_distance_matrices, 'encodings': self.permuted_encodings,'labels': self.permuted_labels, 
            'identifiers': self.permuted_identifiers, 'sources': [self.source] * len(self.permuted_distance_matrices), 
            'coordinating_resis': self.permuted_coordination_labels,
            'metal_coordinates': [self.metal_coords] * len(self.permuted_distance_matrices),})
            df.to_pickle(os.path.join(output_dir, filename))

        else:
            df = pd.DataFrame({'distance_matrices': [self.distance_matrix], 'encodings': [self.encoding], 'labels': [self.label], 
            'identifiers': [self.identifiers], 'sources': [self.source],
            'coordinating_resis': [self.coordination_label],
            'metal_coordinates': [self.metal_coords]})
            df.to_pickle(os.path.join(output_dir, filename))

class CNNCore(Core):
    def __init__(self, source: str, core: AtomGroup, coordinating_resis: np.ndarray, identifiers: list, sequence, channels, putative: bool):
        super().__init__(source, core, coordinating_resis, identifiers, sequence, putative)
        self.distance_channels = channels
        self.channels = self._compute_channels()
        self.permuted_channels = [None]

    def _compute_seq_channels(self, sequence: list):
        """
        Code adapted from https://github.com/lonelu/Metalprot_learning/blob/main/src/extractor/make_bb_info_mats.py
        """
        threelettercodes = {'ALA': 0 , 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'CSO': 4,'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 
                'LYS': 11, 'MET': 12, 'MSE': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'SEP': 15, 'THR': 16, 'TPO': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}

        seq_channels = np.zeros([12, 12, 40], dtype=int)
        for ind, AA in enumerate(sequence):
            if AA not in threelettercodes.keys():
                raise EncodingError

            idx = threelettercodes[AA]
            for j in range(12):
                seq_channels[ind][j][idx] = 1 # horizontal rows of 1's in first 20 channels
                seq_channels[j][ind][idx+20] = 1 # vertical columns of 1's in next 20 channels
        return np.stack([seq_channels[:, :, i] for i in range(40)], axis=0)

    @staticmethod
    def _transform_distance_data(distance_channels: np.ndarray):
        """ 
        Code adapted from dataprocess.py to transform the distance channels into final input.
        """
        #perform transformation on distance channels
        distance_channels[distance_channels == 0] = 20
        distance_channels = (distance_channels - 2) / (18)
        distance_channels = 1 - distance_channels
        
        #fill diagonals with 1s
        _distance_channels = []
        for channel in distance_channels.copy():
            np.fill_diagonal(channel, 1)
            _distance_channels.append(channel)
        return np.stack(_distance_channels, axis=0)

    def _compute_channels(self):
        channels = np.zeros((44,12,12))
        m, n = self.distance_channels[0].shape
        channels[0:4, 0:m, 0:n] = np.stack(self.distance_channels, axis=0)        
        channels[0:4] = CNNCore._transform_distance_data(channels[0:4])
        
        #compute sequence channels
        seq_channels = self._compute_seq_channels(self.sequence)
        channels[4:, 0:12, 0:12] = seq_channels
        return channels

    def _permute_channels(self, fragment_index_permutation, sequence):
        permuted_channel =np.zeros(self.channels.shape)
        for i, I in enumerate(fragment_index_permutation):
            for j, J in enumerate(fragment_index_permutation):
                permuted_channel[0:4,I,J] = self.channels[0:4,i,j]
        permuted_channel[4:,:,:] = self._compute_seq_channels([sequence[i] for i in fragment_index_permutation])
        return permuted_channel

    def permute(self):
        permuted_channels, permuted_labels, permuted_identifiers, permuted_coordinating_resis = [], [], [], []
        fragment_indices = self._identify_fragments()
        sequence = self.structure.select('protein').select('name N').getResnames()
        fragment_permutations = itertools.permutations(list(range(0,len(fragment_indices)))) #get permutations of fragment indices
        no_atoms = len(self.structure.select('protein').select('name N CA C O'))
        atom_indices = np.split(np.linspace(0, no_atoms - 1, no_atoms), no_atoms / 4)
        for permutation in fragment_permutations:
            fragment_index_permutation = sum([fragment_indices[i] for i in permutation], [])

            permuted_channels.append(self._permute_channels(fragment_index_permutation, sequence))
            permuted_labels.append(self._permute_labels(permutation, fragment_indices, atom_indices))
            permuted_identifiers.append([self.identifiers[i] for i in fragment_index_permutation])
            permuted_coordinating_resis.append([self.coordination_label[i] for i in fragment_index_permutation])

        self.permuted_channels = permuted_channels
        self.permuted_labels = permuted_labels
        self.permuted_identifiers = permuted_identifiers
        self.permuted_coordination_labels = permuted_coordinating_resis

    def write_data_files(self, output_dir: str):
        suffix = '_CNN_features.pkl'
        filename = self.filename + suffix
        if self.permuted_channels and self.permuted_labels and self.permuted_identifiers and self.permuted_coordination_labels:
            df = pd.DataFrame({'channels': self.permuted_channels, 'labels': self.permuted_labels, 
            'identifiers': self.permuted_identifiers, 'sources': [self.source] * len(self.permuted_channels), 
            'coordinating_resis': self.permuted_coordination_labels,
            'metal_coords': [self.metal_coords] * len(self.permuted_channels)})
            df.to_pickle(os.path.join(output_dir, filename))

        else:
            df = pd.DataFrame({'channels': [self.channels], 'labels': [self.label], 
            'identifiers': [self.identifiers], 'sources': [self.source],
            'coordinating_resis': [self.coordination_label],
            'metal_coords': [self.metal_coords]})
            df.to_pickle(os.path.join(output_dir, filename))

class Protein:
    def __init__(self, pdb_file: str, cbeta=False):
        self.filepath, self.cbeta = pdb_file, cbeta
        self.structure, self.protein, self.backbone, self.sequence, self._resindices, self._resnums, self._chids, self._no_resis, self._break_positions = self._clean_pdb()
        self.distance_matrix = buildDistMatrix(self.backbone.select('name N C CA O CB')) if cbeta else buildDistMatrix(self.backbone.select('name N C CA O'))
        self.channels = np.stack([buildDistMatrix(self.backbone.select('name N')), buildDistMatrix(self.backbone.select('name CA')), buildDistMatrix(self.backbone.select('name C')), buildDistMatrix(self.backbone.select('name CB'))] , axis=0)
        self.resind2id = dict([(resindex, (resnum, chid)) for resindex, resnum, chid in zip(self._resindices, self._resnums, self._chids)])
        self._connectivity_matrix = self._build_connectivity_matrix()

    @staticmethod
    def _create_atom_group(name: str, attributes: tuple):
        atomgroup = AtomGroup(name)
        coords, names, resnums, resnames, chids = attributes
        atomgroup.setCoords(coords), atomgroup.setNames(names), atomgroup.setResnums(resnums), atomgroup.setResnames(resnames), atomgroup.setChids(chids)
        return atomgroup

    @staticmethod
    def _count_atoms(protein: AtomGroup, resindex: int, atom_name: str):
        atom_sele = protein.select(f'resindex {resindex} and name {atom_name}')
        return len(atom_sele) if atom_sele != None else 0

    @staticmethod
    def _remove_dirty_residues(protein: AtomGroup):
        _N, _CA, _C, _O = protein.select('name N'), protein.select('name CA'), protein.select('name C'), protein.select('name O')
        resindices = set(list(protein.getResindices()))
        if set([len(atom) for atom in [_N, _CA, _C, _O]]) != set([len(resindices)]):
            resindices = list(resindices)
            resindices.sort()
            N_counts, C_counts, CA_counts, O_counts = np.array([]), np.array([]), np.array([]), np.array([])
            for ind in resindices:
                N_counts = np.append(N_counts, Protein._count_atoms(protein, ind, 'N'))
                C_counts = np.append(C_counts, Protein._count_atoms(protein, ind, 'C'))
                CA_counts = np.append(CA_counts, Protein._count_atoms(protein, ind, 'CA'))
                O_counts = np.append(O_counts, Protein._count_atoms(protein, ind, 'O'))
            dirty_indices = list(set(np.where(N_counts != 1)[0].tolist()).union(set(np.where(C_counts != 1)[0].tolist())).union(set(np.where(CA_counts != 1)[0].tolist())).union(set(np.where(O_counts != 1)[0].tolist())))
            _protein = protein.select('not resindex ' + ' '.join(list(map(str, dirty_indices))))
            protein = Protein._create_atom_group('protein', (_protein.getCoords(), _protein.getNames(), _protein.getResnums(), _protein.getResnames(), _protein.getChids()))
            dirty_indices = [i - 1 for i in dirty_indices]
        else:
            dirty_indices = []
        return protein, dirty_indices

    @staticmethod
    def _generate_backbone(protein: AtomGroup):
        _backbone = protein.select('name N CA C O')
        _N, _CA, _C, _O = _backbone.select('name N'), _backbone.select('name CA'), _backbone.select('name C'), _backbone.select('name O')
        N, CA, C = _N.getCoords(), _CA.getCoords(), _C.getCoords()
        CB = _impute_ca_cb(N, CA, C) + CA
        coords = np.vstack([_backbone.getCoords(), CB])
        names = np.concatenate([_backbone.getNames(), np.array(['CB'] * len(CB))])
        resnums = np.concatenate([_backbone.getResnums(), _N.getResnums()])
        resnames = np.concatenate([_backbone.getResnames(), _N.getResnames()])
        chids = np.concatenate([_backbone.getChids(), _N.getChids()])
        backbone = Protein._create_atom_group('backbone', (coords, names, resnums, resnames, chids))
        return backbone, _N.getResnames(), _N.getResindices(), _N.getResnums(), _N.getChids(), len(_N)

    def _clean_pdb(self):
        #identify atoms associated with the protein
        structure = parsePDB(self.filepath)
        termini = structure.select('pdbter').getResindices()
        if len(termini) > 0: 
            protein_resindices = structure.select('protein and not water and not hetero').select('resindex ' + ' '.join(list(map(str, np.arange(0, termini[-1]))))).select('name N').getResindices() 
        else: 
            protein_resindices = structure.select('protein and not water and not hetero').select('name N').getResindices()
        _protein = structure.select('resindex ' + ' '.join(list(map(str, protein_resindices))))
        protein_dirty = Protein._create_atom_group('protein', (_protein.getCoords(), _protein.getNames(), _protein.getResnums(), _protein.getResnames(), _protein.getChids()))
        protein_clean, break_positions = Protein._remove_dirty_residues(protein_dirty)
        backbone, sequence, resindices, resnums, chids, no_resis = Protein._generate_backbone(protein_clean)
        return structure, protein_clean, backbone, sequence, resindices, resnums, chids, no_resis, break_positions

    def _build_connectivity_matrix(self):
        connectivity_matrix = np.zeros((self._no_resis, self._no_resis))
        termini = [max(self.protein.select(f'chid {chid}').getResindices()) for chid in self._chids] + self._break_positions
        termini.sort()
        for ind in self._resindices: 
            if ind in termini:
                continue
            else:
                connectivity_matrix[ind, ind+1], connectivity_matrix[ind+1, ind] = 1, 1
        return connectivity_matrix

    def _get_neighbors(self, coordinating_resind: int, no_neighbors: int):
        """
        Helper function for getting neighboring residues. If a terminal residue is passed, only
        one neighrbor will be returned.
        """
        try:
            fragment = []
            if coordinating_resind == 0:
                neighbor2 = self._connectivity_matrix[coordinating_resind, coordinating_resind+1]
                fragment.append(coordinating_resind)
                fragment.append(coordinating_resind+1) if neighbor2 == 1 else None
            else:
                neighbor1, neighbor2 = self._connectivity_matrix[coordinating_resind, coordinating_resind-1], self._connectivity_matrix[coordinating_resind, coordinating_resind+1]
                fragment.append(coordinating_resind-1) if neighbor1 == 1 else None
                fragment.append(coordinating_resind)
                fragment.append(coordinating_resind+1) if neighbor2 == 1 else None

        except:
            fragment = []
        return fragment 

    def enumerate_cores(self, fcn: bool, cnn: bool, no_neighbors=1, cutoff=15, coordination_number=(2,4)):
        edge_list = []
        putative_coordinating_resis = self.structure.select('protein').select('name CA').select('resname HIS CYS ASP GLU')
        #conditional loop: if there are three sites with His, Cys, etc, we can further look for carbonyl, carboxy etc
        putative_coordinating_resindices = putative_coordinating_resis.getResindices()
        dist_mat = buildDistMatrix(putative_coordinating_resis)
        edge_weights = np.array([])
        row_indexer = 0
        for col_ind in range(len(dist_mat)):
            for row_ind in range(1+row_indexer, len(dist_mat)):
                distance = dist_mat[row_ind, col_ind]
                if distance <= cutoff:
                    edge_list.append(np.array([putative_coordinating_resindices[col_ind], putative_coordinating_resindices[row_ind]]))
                    edge_weights = np.append(edge_weights, distance)
            row_indexer += 1
        edge_list = Protein._filter_by_angle(np.vstack(edge_list), self.structure, edge_weights)
        cliques = enumerateCliques(np.array(edge_list), coordination_number[1])[coordination_number[0]:]

        max_atoms = 12 * 4 if not self.cbeta else 12 * 5
        fcn_cores, cnn_cores = self._construct_cores(cliques, max_atoms, no_neighbors, cnn, fcn)
        return fcn_cores, cnn_cores

    def _construct_cores(self, cliques, max_atoms: int, no_neighbors: int, fcn: bool, cnn: bool):
        fcn_cores, cnn_cores = [], []
        no_resis = len(set(self.backbone.getResindices()))
        splits = sum([np.vsplit(x, no_resis) for x in np.hsplit(self.distance_matrix, no_resis)], [])
        combinations = [(j,i) for i in range(0, no_resis) for j in range(0, no_resis)]
        split_mapper = dict([(combination, splits[ind]) for combination, ind in zip(combinations, range(len(combinations)))])

        for subclique in cliques:
            for clique in subclique:
                binding_core = np.sort(np.array(list(set(sum([self._get_neighbors(resind, no_neighbors) for resind in list(clique)], [])))))
                # print(len(self.sequence), clique, binding_core, self._resindices[-1])
                sequence = self.sequence[binding_core]
                identifiers = [self.resind2id[resind] for resind in binding_core]
                core = self.structure.select('resindex ' + ' '.join([str(num) for num in binding_core]))
                # print(core)

                if fcn:
                    combinations = list(itertools.product(*[binding_core, binding_core]))
                    sub_matrices = np.array([split_mapper[combination] for combination in combinations])
                    m, n, r = sub_matrices.shape
                    matrix = sub_matrices.reshape(-1,len(binding_core),n,r).transpose(0,2,1,3).reshape(-1,len(binding_core)*r)
                    padded = np.zeros((max_atoms, max_atoms))
                    padded[0:len(matrix), 0:len(matrix)] = matrix
                    fcn_cores.append(FCNCore(self.filepath, core, clique, identifiers, sequence, padded, putative=True))

                if cnn:
                    binding_core = np.array(binding_core)
                    distance_matrix_channels = np.zeros((4,12,12))
                    row_inds, col_inds = np.meshgrid(binding_core, binding_core)
                    distance_matrix_channels = self.channels[:, row_inds, col_inds]
                    cnn_cores.append(CNNCore(self.filepath, core, clique, identifiers, sequence, distance_matrix_channels, putative=True))
        return fcn_cores, cnn_cores

    @staticmethod
    def _filter_by_angle(edge_list: np.ndarray, structure: AtomGroup, distances: np.ndarray):
        """Filters pairs of contacts based on relative orientation of Ca-Cb and Ca-Ca bond vectors. 

        Args:
            edge_list (np.ndarray): nx2 array containing pairs of contacts.
            structure (AtomGroup): AtomGroup object of input structure.
            distances (np.ndarray): Array of length n containing distance between each contact.

        Returns:
            filtered (np.ndarray): nx2 array containing filtered contacts.
        """

        #get backbone atom coordinates for all residues included in the edge list
        all_resindices = set(np.concatenate(list(edge_list)))
        coordinates = dict([(resindex, structure.select('protein').select('name C CA N').select(f'resindex {resindex}').getCoords()) for resindex in all_resindices])

        #for each pair of contacts, get coordinates for atom i and j
        n_i, n_j = np.vstack([coordinates[resindex][0].flatten() for resindex in edge_list[:,0]]), np.vstack([coordinates[resindex][0].flatten() for resindex in edge_list[:,1]])
        ca_i, ca_j = np.vstack([coordinates[resindex][1].flatten() for resindex in edge_list[:,0]]), np.vstack([coordinates[resindex][1].flatten() for resindex in edge_list[:,1]])
        c_i, c_j = np.vstack([coordinates[resindex][2].flatten() for resindex in edge_list[:,0]]), np.vstack([coordinates[resindex][2].flatten() for resindex in edge_list[:,1]])

        #compute ca-cb bond vector for atom i and j
        ca_cb_i, ca_cb_j = _impute_ca_cb(n_i, ca_i, c_i), _impute_ca_cb(n_j, ca_j, c_j)

        #compute the ca-ca bond vector between atom i and j. also compute the ca-ca/ca-cbi and ca-ca/ca-bj angles.
        ca_i_ca_j = ca_j - ca_i
        angles_i, angles_j = Protein._compute_angles(ca_cb_i, ca_i_ca_j), Protein._compute_angles(ca_cb_j, ca_i_ca_j)

        #filter based on angle cutoffs
        accepted = np.argwhere(distances <= 7)
        filtered_inds = np.intersect1d(np.intersect1d(np.argwhere(angles_i < 130), np.argwhere(angles_j > 30)), np.argwhere(distances > 7))
        filtered = edge_list[np.union1d(accepted, filtered_inds)]
        return filtered

    @staticmethod
    def _compute_angles(vec1: np.ndarray, vec2: np.ndarray):
        """Helper function for computing angles between bond vectors in a vectorized fashion.

        Args:
            vec1 (np.ndarray): An nx3 array containing bond vectors.
            vec2 (np.ndarray): Another nx3 array containing bond vectors.

        Returns:
            angles (np.ndarray): The angles between the vectors. 
        """

        dot = np.sum(vec1 * vec2, axis=1)
        norm1, norm2 = np.linalg.norm(vec1, axis=1), np.linalg.norm(vec2, axis=1)
        angles = np.degrees(np.arccos(dot / (norm1 * norm2)))
        return angles

class MetalloProtein(Protein):
    def __init__(self, pdb_file: str, cbeta=False):
        super().__init__(pdb_file, cbeta)
        self.metalloprotein = self._process_metalloprotein(self.structure.select('hetero').select('name ZN'))
        self.resind2id = dict([(resindex, (resnum, chid)) for resindex, resnum, chid in zip(self.metalloprotein.getResindices(), self.metalloprotein.getResnums(), self.metalloprotein.getChids())])

    def _process_metalloprotein(self, metals: AtomGroup):
        coords = np.vstack([self.protein.getCoords(), metals.getCoords()])
        names = np.concatenate([self.protein.getNames(), metals.getNames()])
        resnums = np.concatenate([self.protein.getResnums(), metals.getResnums()])
        resnames = np.concatenate([self.protein.getResnames(), metals.getResnames()])
        chids = np.concatenate([self.protein.getChids(), metals.getChids()])
        metalloprotein = Protein._create_atom_group('metalloprotein', (coords, names, resnums, resnames, chids))
        return metalloprotein

    def enumerate_cores(self, cnn: bool, fcn: bool, no_neighbors=1, coordination_number=(2,4)):
        """
        Extracts metal binding cores from an input protein structure. Returns a list of
        core objects.
        """
        cliques = []
        metal_inds = []
        for metal_ind in self.metalloprotein.select('name ZN').getResindices():
            try: #try/except to account for solvating metal ions included for structure determination
                coordinating_resindices = list(set(self.metalloprotein.select(f'protein and not carbon and not hydrogen and within 2.83 of resindex {metal_ind}').getResindices()))
            except:
                continue
            
            if len(coordinating_resindices) <= coordination_number[1] and len(coordinating_resindices) >= coordination_number[0]:
                cliques.append(coordinating_resindices)
                metal_inds.append(metal_ind)
            else:
                continue
        max_atoms = 12 * 5 if self.cbeta else 12 * 4
        fcn_cores, cnn_cores = self._construct_cores(cliques, max_atoms, no_neighbors, fcn, cnn, metal_inds)
        return fcn_cores, cnn_cores

    def _construct_cores(self, cliques, max_atoms: int, no_neighbors: int, fcn: bool, cnn: bool, metal_inds: list):
        fcn_cores, cnn_cores = [], []
        splits = sum([np.vsplit(x, self._no_resis) for x in np.hsplit(self.distance_matrix, self._no_resis)], [])
        combinations = [(j,i) for i in range(0, self._no_resis) for j in range(0, self._no_resis)]
        split_mapper = dict([(combination, splits[ind]) for combination, ind in zip(combinations, range(len(combinations)))])
        for clique, metal_ind in zip(cliques, metal_inds):
            binding_core = np.sort(np.array(list(set(sum([self._get_neighbors(resind, no_neighbors) for resind in list(clique)], [])))))
            sequence = self.sequence[binding_core]
            identifiers = [self.resind2id[resind] for resind in binding_core]
            core = self.metalloprotein.select('resindex ' + ' '.join([str(num) for num in np.append(binding_core, metal_ind)]))

            # print(core.getResnums(), core.getResnames())

            if fcn:
                combinations = list(itertools.product(*[binding_core, binding_core]))
                sub_matrices = np.array([split_mapper[combination] for combination in combinations])
                m, n, r = sub_matrices.shape
                matrix = sub_matrices.reshape(-1,len(binding_core),n,r).transpose(0,2,1,3).reshape(-1,len(binding_core)*r)
                padded = np.zeros((max_atoms, max_atoms))
                padded[0:len(matrix), 0:len(matrix)] = matrix
                fcn_cores.append(FCNCore(self.filepath, core, clique, identifiers, sequence, padded, putative=False))

            if cnn:
                binding_core = np.array(binding_core)
                distance_matrix_channels = np.zeros((4,12,12))
                row_inds, col_inds = np.meshgrid(binding_core, binding_core)
                distance_matrix_channels = self.channels[:, row_inds, col_inds]
                cnn_cores.append(CNNCore(self.filepath, core, clique, identifiers, sequence, distance_matrix_channels, putative=False))
        return fcn_cores, cnn_cores

