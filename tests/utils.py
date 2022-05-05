import numpy as np

def get_coo_arrays(n_verts, faces, lambda_):

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
    adj_values = np.ones(adj.shape[1], dtype=np.float64) * lambda_

    # Diagonal indices, duplicated as many times as the connectivity of each index
    diag_idx = np.stack((adj[0], adj[0]), axis=0)

    diag = np.stack((np.arange(n_verts), np.arange(n_verts)), axis=0)

    # Build the sparse matrix
    idx = np.concatenate((adj, diag_idx, diag), axis=1)
    values = np.concatenate((-adj_values, adj_values, np.ones(n_verts)))

    return values, idx

# https://sinestesia.co/blog/tutorials/python-icospheres/
def get_icosphere(level=0):
    # Generate the base, that we then subdivide
    n_verts = 12

    faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],

            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],

            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],

            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]]

    split_cache = {}

    def split(i, j):
        key = f"{min(i,j)}-{max(i,j)}"

        if key in split_cache:
            return split_cache[key]

        split_cache[key] = n_verts

        return n_verts

    for i in range(level):
        faces_subdiv = []

        for tri in faces:
            v1 = split(tri[0], tri[1])
            n_verts = max(n_verts, v1+1)
            v2 = split(tri[1], tri[2])
            n_verts = max(n_verts, v2+1)
            v3 = split(tri[2], tri[0])
            n_verts = max(n_verts, v3+1)

            faces_subdiv.append([tri[0], v1, v3])
            faces_subdiv.append([tri[1], v2, v1])
            faces_subdiv.append([tri[2], v3, v2])
            faces_subdiv.append([v1, v2, v3])

        faces = faces_subdiv

    return n_verts, np.array(faces)

def get_cube():
    faces = np.array([[0, 1, 3],
                     [0, 3, 2],
                     [2, 3, 7],
                     [2, 7, 6],
                     [4, 5, 7],
                     [4, 7, 6],
                     [0, 1, 5],
                     [0, 5, 4],
                     [1, 5, 7],
                     [1, 7, 3],
                     [0, 4, 6],
                     [0, 6, 2]])
    return 8, faces
