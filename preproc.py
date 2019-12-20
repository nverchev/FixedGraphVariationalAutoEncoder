
import numpy as np
import itertools
import scipy.sparse as sparse

def dist(V, F):
    num_vertices = V.shape[0]
    W = np.zeros((num_vertices, num_vertices))

    for face in F:
        vertices = face.tolist()
        for i, j in itertools.product(vertices, vertices):
            W[i, j] = np.sqrt(((V[i] - V[j]) ** 2).sum())

    return sparse.csr_matrix(W)

def area(F, l):
    areas = np.zeros(F.shape[0])

    for f in range(F.shape[0]):
        i, j, k = F[f].tolist()
        sijk = (l[i, j] + l[j, k] + l[k, i]) / 2
        sum_ = max(sijk * (sijk - l[i, j]) * (sijk - l[j, k]) * (sijk - l[k, i]),1e-12)
        areas[f] = np.sqrt(sum_)
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
    # assert np.abs(L - L.T).mean() < 1e-9
    return D - W

def quaternion_matrix(x):
    a, b, c, d = x.tolist()
    return np.array([[a, -b, -c, -d],
                     [b, a, -d, c],
                     [c, d, a, -b],
                     [d, -c, b, a]])

def dirac(V, F, l, Af):
    Av = np.zeros(V.shape[0])
    Di = np.zeros((4 * F.shape[0], 4 * V.shape[0]))
    DiA = np.zeros((4 * V.shape[0], 4 * F.shape[0]))
    simple_Di = np.zeros((4 * F.shape[0], 4 * V.shape[0]))
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
            simple_Di[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = mat
            Di[i * 4:(i + 1) * 4, j * 4: (j + 1) * 4] = mat / Af[i]
            DiA[j * 4:(j + 1) * 4, i * 4: (i + 1) * 4] = -mat / Av[j]
    simple_Di = sparse.csr_matrix(simple_Di)
    Di = sparse.csr_matrix(Di)
    DiA = sparse.csr_matrix(DiA)

    return Di, DiA ,simple_Di

def process(sample):
    V, F = sample
    l = dist(V, F)
    a = area(F, l)
    W, A = cotangent_weights(F, a, l)
    L = laplacian(W, A)
    L_norm=A*L
    Di, DiA,simple_Di = dirac(V, F, l, a)
    for array in [V, F, L.data, L_norm.data,Di.data, DiA.data,simple_Di.data]:
        assert not np.isnan(array).any()
    return V, L, L_norm,Di, DiA,simple_Di

