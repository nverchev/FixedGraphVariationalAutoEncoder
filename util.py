import torch
import numpy as np
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