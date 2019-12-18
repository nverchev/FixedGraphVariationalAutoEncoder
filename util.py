import torch
import numpy as np

import itertools
import scipy.sparse as sparse





def sparse_diag_cat(tensors, size0, size1):
    values = []
    for i, tensor in enumerate(tensors):
        values.append(tensor._values())
    indices = []
     # assuming COO
    for i, t in enumerate(tensors):
        indices.append(t._indices()+i*torch.LongTensor([[size0], [size1]]))
    values = torch.cat(values, 0)
    indices = torch.cat(indices, 1)
    size = torch.Size((len(tensors)*size0, len(tensors)*size1))
    return torch.sparse.FloatTensor(indices, values, size).coalesce()


# @title mean operator
def pt_to_pt_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def quaternion_matrix(x):
    a, b, c, d = x.tolist()
    return np.array([[a, -b, -c, -d],
                     [b, a, -d, c],
                     [c, d, a, -b],
                     [d, -c, b, a]])


def simple_dirac(V, F):
    D = np.zeros((4 * F.shape[0], 4 * V.shape[0]))
    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            ind1 = F[i, (ind + 1) % 3]
            ind2 = F[i, (ind + 2) % 3]
            e1 = V[ind1]
            e2 = V[ind2]
            e = np.array([0, e1[0] - e2[0], e1[1] - e2[1], e1[2] - e2[2]])
            D[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = -quaternion_matrix(e)
    D = pt_to_pt_sparse(torch.tensor(D))
    return D


def sp_sparse_to_pt_sparse(L):
    """
    Converts a scipy matrix into a pytorch one.
    """
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    indices = torch.from_numpy(indices).long()
    L_data = torch.from_numpy(L.data)

    size = torch.Size(L.shape)
    indices = indices.transpose(1, 0)

    L = torch.sparse.FloatTensor(indices, L_data, size)
    return L


def dist(V, F):
    num_vertices = V.shape[0]
    W = np.zeros((num_vertices, num_vertices))

    for face in F:
        vertices = face.tolist()
        for i, j in itertools.product(vertices, vertices):
            W[i, j] = np.sqrt(((V[i] - V[j]) ** 2).sum())

    return sparse.csr_matrix(W)


def dirac(V, F, l, Af):
    Av = np.zeros(V.shape[0])
    D = np.zeros((4 * F.shape[0], 4 * V.shape[0]))
    DA = np.zeros((4 * V.shape[0], 4 * F.shape[0]))

    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            Av[j] += Af[i] / 3

    for i in range(F.shape[0]):
        for ind, j in enumerate(F[i]):
            ind1 = F[i, (ind + 1) % 3]
            ind2 = F[i, (ind + 2) % 3]

            e1 = V[ind1]
            e2 = V[ind2]

            e = np.array([0, e1[0] - e2[0], e1[1] - e2[1], e1[2] - e2[2]])

            mat = -quaternion_matrix(e) / 2
            D[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = mat / Af[i]
            DA[j * 4:(j + 1) * 4, i * 4: (i + 1) * 4] = -mat / Av[j]
    D = pt_to_pt_sparse(torch.tensor(D))
    DA = pt_to_pt_sparse(torch.tensor(DA))

    return D, DA


def area(F, l):
    areas = np.zeros(F.shape[0])

    for f in range(F.shape[0]):
        i, j, k = F[f].tolist()
        sijk = (l[i, j] + l[j, k] + l[k, i]) / 2
        sum_ = sijk * (sijk - l[i, j]) * (sijk - l[j, k]) * (sijk - l[k, i])
        areas[f] = min(np.sqrt(sum_), 1e-6)
    return areas


def cotangent_weights(F, a, l):
    W = np.zeros(l.shape)
    A = np.zeros(l.shape[0])

    for f in range(F.shape[0]):
        for v_ind in itertools.permutations(F[f].tolist()):
            i, j, k = v_ind
            W[i, j] += (-l[i, j] ** 2 + l[j, k] ** 2 + l[k, i] ** 2) / (8 * a[f] + 1e-5)
            A[i] += a[f] / 3 / 2  # each face will appear 2 times

    return sparse.csr_matrix(W), sparse.diags(1 / (A + 1e-5), 0)


def laplacian(W, A_inv):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)
    D = sparse.diags(d.A.squeeze(), 0)
    L = A_inv * (D - W)
    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sparse.csr.csr_matrix

    return sp_sparse_to_pt_sparse(L)




def process(sample):
    V, F = sample
    l = dist(V, F)
    a = area(F, l)
    W, A = cotangent_weights(F, a, l)
    L = laplacian(W, A)
    Di, DiA = dirac(V, F, l, a)
    simple_Di= simple_dirac(V, F)
    return L.float(), Di.float(), DiA.float(),simple_Di.float()
