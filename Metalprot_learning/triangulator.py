import sys

import numpy as np
import numba as nb
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

@nb.njit
def prepare_distmats_and_verts(pred_distances, distance_matrix, coords):
    """Prepare distance matrices and Cartesian vertex coordinates per core.

    Parameters
    ----------
    pred_distances : np.array [48]
        Array of predicted distances of the metal ion from the core atoms.
    distance_matrix : np.array [48 x 48]
        Distance matrix for the core atoms.
    coords : np.array [48 x 3]
        Cartesian coordinates of the core atoms.
    Returns
    -------
    tetrahedron_distmats : np.array [n_combinations x 5 x 5]
        Distance matrices for each tetrahedron of three core atoms and 
        the metal, padded with an initial row/column of 1's (and 0 on the 
        diagonal).  The determinant of the square of each matrix is the 
        Cayley-Menger determinant instrumental in distance geometry.
    triangle_verts : np.array [n_combinations x 3 x 3]
        Cartesian coordinates of the three core atoms in each tetrahedron.
    """
    n_nonzero_preds = np.sum(pred_distances > 0.)
    n_combinations = n_nonzero_preds * \
                     (n_nonzero_preds - 1) * \
                     (n_nonzero_preds - 2) // 6
    tetrahedron_distmats = np.ones((n_combinations, 5, 5)) - np.eye(5)
    triangle_verts = np.empty((n_combinations, 3, 3))
    counter = 0
    for i in range(n_nonzero_preds - 2):
        for j in range(i + 1, n_nonzero_preds - 1):
            for k in range(j + 1, n_nonzero_preds):
                tetrahedron_distmats[counter, 1, 2] = distance_matrix[i, j]
                tetrahedron_distmats[counter, 2, 1] = distance_matrix[j, i]
                tetrahedron_distmats[counter, 2, 3] = distance_matrix[j, k]
                tetrahedron_distmats[counter, 3, 2] = distance_matrix[k, j]
                tetrahedron_distmats[counter, 3, 1] = distance_matrix[k, i]
                tetrahedron_distmats[counter, 1, 3] = distance_matrix[i, k]
                tetrahedron_distmats[counter, 1, 4] = pred_distances[i]
                tetrahedron_distmats[counter, 4, 1] = pred_distances[i]
                tetrahedron_distmats[counter, 2, 4] = pred_distances[j]
                tetrahedron_distmats[counter, 4, 2] = pred_distances[j]
                tetrahedron_distmats[counter, 3, 4] = pred_distances[k]
                tetrahedron_distmats[counter, 4, 3] = pred_distances[k]
                triangle_verts[counter, 0] = coords[i] # r0
                triangle_verts[counter, 1] = coords[j] # r1
                triangle_verts[counter, 2] = coords[k] # r2
                counter += 1
    return tetrahedron_distmats, triangle_verts

def triangulate(pred_distances, distance_matrix, coords):
    """Use distance geometry to triangulate the metal from core distances.

    Parameters
    ----------
    pred_distances : np.array [48]
        Array of predicted distances of the metal ion from the core atoms.
    distance_matrix : np.array [48 x 48]
        Distance matrix for the core atoms.
    coords : np.array [48 x 3]
        Cartesian coordinates of the core atoms.

    Returns
    -------
    metal_pred : np.array [3]
        Predicted Cartesian coordinates of the metal, defined as the mean 
        of the point cloud determined from distance geometry considerations 
        of all trimers of core atoms.
    uncertainty : float
        Uncertainty in the predicted coordinates, defined as the RMSD of 
        the point cloud from the predicted metal location.
    """
    # if np.sum(pred_distances > 0.) not in [24, 36, 48]:
    #     raise ValueError(('The number of nonzero predicted distances is '
    #                       'not 24, 36, or 48.'))
    tetrahedron_distmats, triangle_verts = \
        prepare_distmats_and_verts(pred_distances, distance_matrix, coords)
    # compute local orthonormal frames for each triangle of core atoms
    v1 = triangle_verts[:, 1] - triangle_verts[:, 0]
    e1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = triangle_verts[:, 2] - triangle_verts[:, 0]
    v2 -= np.sum(v2 * e1, axis=1, keepdims=True) * e1
    e2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    e3 = np.cross(e1, e2, axis=1)
    # compute area of triangular base formed by each triplet of pair atoms
    A = np.sqrt(np.abs(np.linalg.det(tetrahedron_distmats[:, :4, :4] ** 2)) 
                / 16.)
    # compute volume of tetrahedron formed by each triplet and the predicted 
    # metal distances
    V = np.sqrt(np.abs(np.linalg.det(tetrahedron_distmats ** 2)) / 288.)
    # compute relevant coordinates in the local frames
    x2 = (tetrahedron_distmats[:, 1, 2] ** 2 + 
          tetrahedron_distmats[:, 1, 3] ** 2 - 
          tetrahedron_distmats[:, 2, 3] ** 2) / \
         tetrahedron_distmats[:, 1, 2] / 2.
    y2 = 2. * A / tetrahedron_distmats[:, 1, 2]
    x3 = (tetrahedron_distmats[:, 1, 2] ** 2 + 
          tetrahedron_distmats[:, 1, 4] ** 2 - 
          tetrahedron_distmats[:, 2, 4] ** 2) / \
         tetrahedron_distmats[:, 1, 2] / 2.
    y3 = (tetrahedron_distmats[:, 1, 3] ** 2 + 
          tetrahedron_distmats[:, 1, 4] ** 2 - 
          tetrahedron_distmats[:, 3, 4] ** 2 - 
          2. * x2 * x3) / y2 / 2.
    z3 = 3. * V / A
    r3_xy = triangle_verts[:, 0] + \
            x3.reshape((-1, 1)) * e1 + \
            y3.reshape((-1, 1)) * e2
    r3_pos = r3_xy + z3.reshape((-1, 1)) * e3
    r3_neg = r3_xy - z3.reshape((-1, 1)) * e3

    # use DBSCAN clustering to find the largest cluster of metal predictions
    all_metal_coords = np.empty((2 * len(r3_pos), 3))
    all_metal_coords[::2] = r3_pos
    all_metal_coords[1::2] = r3_neg

    # metal_pred_dists = cdist(all_metal_coords, all_metal_coords)
    clustering = DBSCAN(eps=0.3, min_samples=4).fit(all_metal_coords)
    if np.all(clustering.labels_ == -1):
        raise ValueError('Clustering of point cloud failed.')
    largest = mode(clustering.labels_[clustering.labels_ >= 0]).mode
    largest_mask = (clustering.labels_ == largest)
    centroid = np.mean(all_metal_coords[largest_mask], axis=0, keepdims=True)
    # for each triplet of core atoms, select the prediction that is closest 
    # to the centroid of the largest cluster
    pos_dists = cdist(r3_pos, centroid).flatten()
    neg_dists = cdist(r3_neg, centroid).flatten()
    canonical_metal_coords = np.empty((len(r3_pos), 3))
    canonical_metal_coords[pos_dists <= neg_dists] = \
        r3_pos[pos_dists <= neg_dists]
    canonical_metal_coords[pos_dists > neg_dists] = \
        r3_neg[pos_dists > neg_dists]
    # compute the mean and stdev of the predictions that are closest to the 
    # largest cluster
    metal_coords = np.mean(canonical_metal_coords, axis=0)
    uncertainty = np.sqrt(3. * np.mean(
        (canonical_metal_coords - metal_coords) ** 2))
    return metal_coords, uncertainty
