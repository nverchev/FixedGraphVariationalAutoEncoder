from models import LapVAE, DirVAE, LapVAE_old
from util import sparse_diag_cat, sp_sparse_to_pt_sparse
from losses import *
import torch
import numpy as np
import glob
import numpy as np
import re
import os
from torch import optim
from preproc import process
# Change this
from minio import Minio
from minio.error import ResponseError
import argparse

minioClient = Minio('s3nverchev.ugent.be',
                    access_key='M12AX6PRAW2HUPUHPTL0',
                    secret_key='yC2UlxBD+exlGz5S+zLaVclYUvRQ9D8msgOMVAWh',
                    secure=True)
num_vertices = 5023
num_faces = 9976

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Coma VAE')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-epoch', type=int, default=1000, metavar='N',
                    help='num of training epochs (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--adaptive-cycle', type=int, default=10000, metavar='N',
                    help='epochs before it halves learning rate (default: 10000)')
parser.add_argument('--num-features', type=int, default=64, metavar='N',
                    help='number of features (default: 64)')
parser.add_argument('--num-blocks-encoder', type=int, default=2, metavar='N',
                    help='net architecture (default: 2)')
parser.add_argument('--num-blocks-decoder', type=int, default=3, metavar='N',
                    help='net architecture (default: 3)')
parser.add_argument('--dim-latent', type=int, default=8, metavar='N',
                    help='dimension latent space  (default: 8)')
parser.add_argument('--weight_decay', type=float, default=0.000001, metavar='N',
                    help='regulizer (default: 100)')
parser.add_argument('--initial_learning_rate', type=float, default=0.01, metavar='N',
                    help='learning rate (default: 0.001)')
parser.add_argument('--model', default="lap_adj",
                    help='lap | lap_norm | lap_old| lap_adj | dirac | simple_dirac')
parser.add_argument('--loss', default="ELBO",
                    help='ELBO | L1 | mixed')
parser.add_argument('--version', default="hpc_temp")
parser.add_argument('--load-version', type=int, default=1000, metavar='N',
                    help="-1 don't load,0 most recent epoch")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

operator = args.model
num_epoch = args.num_epoch
batch_size = args.batch_size
num_features = args.num_features
dim_latent = args.dim_latent
num_blocks_encoder = args.num_blocks_encoder
num_blocks_decoder = args.num_blocks_decoder
initial_learning_rate = args.initial_learning_rate
weight_decay = args.weight_decay
learning_factor = args.adaptive_cycle
version = args.version
load = args.load_version
loss_ = args.loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sample_batch(samples, train=True):
    indices = []
    if train:
        for b in range(batch_size):
            ind = np.random.randint(0, len(samples))
            indices.append(ind)
    else:
        indices = list(range(len(samples)))

    inputs = torch.zeros(batch_size, num_vertices, 3)
    faces = torch.zeros(batch_size, num_faces, 3).long()
    laplacian = []
    Di = []
    DiA = []
    for b, ind in enumerate(indices):
        inputs[b] = torch.tensor(samples[ind]['V'])

        if operator == "lap":
            laplacian.append(sp_sparse_to_pt_sparse(samples[ind]['L']))
        if operator == "lap_norm" or operator == 'lap_old':
            laplacian.append(sp_sparse_to_pt_sparse(samples[ind]['L_norm']))
        if operator == "dirac":
            Di.append(sp_sparse_to_pt_sparse(samples[ind]['Di']))
            DiA.append(sp_sparse_to_pt_sparse(samples[ind]['DiA']))
        if operator == "simple_dirac":
            Di.append(sp_sparse_to_pt_sparse(samples[ind]['Di']))
    if operator == "lap" or operator == "lap_norm":
        laplacian = sparse_diag_cat(laplacian, num_vertices, num_vertices)
        return inputs.to(device), laplacian.to(device), None, None
    if operator == "lap_adj":
        return inputs.to(device), None, None, None
    if operator == "dirac":
        Di = sparse_diag_cat(Di, 4 * num_faces, 4 * num_vertices)
        DiA = sparse_diag_cat(DiA, 4 * num_vertices, 4 * num_faces)
        return inputs.to(device), None, Di.to(device), DiA.to(device)
    if operator == "simple_dirac":
        Di = sparse_diag_cat(Di, 4 * num_faces, 4 * num_vertices)
        return inputs.to(device), None, Di.to(device), None

with open("err.txt", 'w') as file:
            file.write(str(1))

mean_shape = torch.tensor(np.load('mean_shape.npy', allow_pickle=True)).to(device)

data = []
labels = []

path_list = sorted(glob.glob("../data_vo/*/V/*"))
path_list.extend(sorted(glob.glob("../scratch_kyukon_vo/*/V/*")))
path_list.extend(sorted(glob.glob("../scratch_phanpy_vo/*/V/*")))
path_list.extend(sorted(glob.glob("../scratch_kyukon/*/V/*")))
path_list.extend(sorted(glob.glob("../scratch_phanpy/*/V/*")))
for i, path in enumerate(path_list):
    for sample in np.load(path):
        data = np.hstack([data, {'V': sample}])
        labels.append(int(path.split('/')[-1][:2]))

if operator == 'lap':
    operator_dir = 'L'
    L = sp_sparse_to_pt_sparse(np.load('mean_L.npy', allow_pickle=True).tolist().astype('f4'))
    mean_L = sparse_diag_cat([L for _ in range(batch_size)], num_vertices, num_vertices).to(device)
elif operator == 'lap_norm' or operator == 'lap_old':
    operator_dir = 'L_norm'
    L_norm = sp_sparse_to_pt_sparse(np.load('mean_L_norm.npy', allow_pickle=True).tolist().astype('f4'))
    mean_L = sparse_diag_cat([L_norm for _ in range(batch_size)], num_vertices, num_vertices).to(device)
elif operator == 'lap_adj':
    operator_dir = 'L_norm'
    L_adj = sp_sparse_to_pt_sparse(np.load('mean_L_norm.npy', allow_pickle=True).tolist().astype('f4'))
    L_adj = sparse_diag_cat([L_adj for _ in range(batch_size)], num_vertices, num_vertices).to(device)
elif operator == 'dirac':
    operator_dir = 'Di'
    Di = sp_sparse_to_pt_sparse(np.load('mean_Di.npy', allow_pickle=True).tolist().astype('f4'))
    mean_Di = sparse_diag_cat([Di for _ in range(batch_size)], 4 * num_faces, 4 * num_vertices).to(device)
    DiA = sp_sparse_to_pt_sparse(np.load('mean_DiA.npy', allow_pickle=True).tolist().astype('f4'))
    mean_DiA = sparse_diag_cat([DiA for _ in range(batch_size)], 4 * num_vertices, 4 * num_faces).to(device)

elif operator == 'simple_dirac':
    operator_dir = 'simple_Di'
    simple_Di = sp_sparse_to_pt_sparse(np.load('mean_simple_Di.npy', allow_pickle=True).tolist().astype('f4'))
    mean_simple_Di = sparse_diag_cat([simple_Di for _ in range(batch_size)], 4 * num_faces, 4 * num_vertices).to(device)

path_list = sorted(glob.glob("../data_vo/*/" + operator_dir + "/*"))
path_list.extend(sorted(glob.glob("../scratch_kyukon_vo/*/" + operator_dir + "/*")))
path_list.extend(sorted(glob.glob("../scratch_phanpy_vo/*/" + operator_dir + "/*")))
path_list.extend(sorted(glob.glob("../scratch_kyukon/*/" + operator_dir + "/*")))
path_list.extend(sorted(glob.glob("../scratch_phanpy/*/" + operator_dir + "/*")))
with open("err.txt", 'w') as file:
    file.write(str(2))
i = 0
for path in path_list:
    print(path)
    for sample in np.load(path):
        data[i][operator_dir] = sample.astype('f4')
        i += 1

if operator == 'dirac':
    path_list = sorted(glob.glob("../data_vo/*/DiA/*"))
    path_list.extend(sorted(glob.glob("../scratch_kyukon_vo/*/DiA/*")))
    path_list.extend(sorted(glob.glob("../scratch_phanpy_vo/*/DiA/*")))
    path_list.extend(sorted(glob.glob("../scratch_kyukon/*/DiA/*")))
    path_list.extend(sorted(glob.glob("../scratch_phanpy/*/DiA/*")))
    i = 0
    for path in path_list:
        print(path)
        for sample in np.load(path):
            data[i]['DiA'] = sample.astype('f4')
            i += 1
with open("err.txt", 'w') as file:
    file.write(str(3))
train_data = []
val_data = []
test_data = []
for i, sample in enumerate(data):
    if i % 10 == 8:
        val_data.append(sample)
    elif i % 10 == 9:
        test_data.append(sample)
    else:
        train_data.append(sample)

test_labels = []
for i, label in enumerate(labels):
    if i % 10 == 9:
        test_labels.append(label)

if operator == "lap" or operator == 'lap_norm' or operator == 'lap_adj':
    model = LapVAE(num_features, num_blocks_encoder, num_blocks_decoder, dim_latent)
elif operator == 'lap_old':
    model = LapVAE_old(num_features, num_blocks_encoder, num_blocks_decoder, dim_latent)
else:
    model = DirVAE(num_features, num_blocks_encoder, num_blocks_decoder, dim_latent)

init_epoch = 1
train_performances = []
val_performances = []
optimizer = optim.Adam(model.parameters(), initial_learning_rate, weight_decay=weight_decay)
with open("err.txt", 'w') as file:
    file.write(str(4))
try:
    os.mkdir(operator + '_' + version)
except:
    pass
if load + 1:
    try:

        for file in minioClient.list_objects('coma', recursive=True):
            file = file.object_name
            if file.split("/")[0] == operator + '_' + version:
                minioClient.fget_object('coma', file, file)
        list_models = {int(re.sub("\D", "", file)): file for file in
                       glob.glob('{}/model_epoch*'.format(operator + '_' + version))}
        list_optimizer = {int(re.sub("\D", "", file)): file for file in
                          glob.glob('{}/optimizer_epoch*'.format(operator + '_' + version))}
        max_epoch = load if load else max(list_models.keys())
        init_epoch = max_epoch + 1
        model.load_state_dict(torch.load(list_models[max_epoch]))
        model.to(device)
        optimizer.load_state_dict(torch.load(list_optimizer[max_epoch]))
        with open("{}/train_performances.txt".format(operator + '_' + version), 'r') as file:
            train_performances = eval(file.read())
        with open("{}/val_performances.txt".format(operator + '_' + version), 'r') as file:
            val_performances = eval(file.read())
        print("loaded: ", list_models[max_epoch])
    except:
        print("No saved models!")
with open("err.txt", 'w') as file:
    file.write(str(5))
num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))
model.cuda()
for epoch in range(init_epoch, init_epoch + num_epoch):
    model.train()
    loss_value = 0.0
    loss_bce = 0.0
    loss_kld = 0.0
    L1_loss = 0
    # Train
    for j in range(len(train_data) // batch_size):
        inputs, L, Di, DiA = sample_batch(train_data)
        if  operator == "lap_old":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L)
        if  operator == "lap_adj":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L_adj, mean_shape, L_adj)
        if operator == "lap" or operator == "lap_norm":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L, mean_shape, mean_L)
        if operator == "dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA, mean_shape, mean_Di, mean_DiA)
        if operator == "simple_dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, Di.transpose(1, 0), mean_shape, mean_simple_Di,
                                                          mean_simple_Di.transpose(1, 0))
        BCE, KLD = loss_function(recon_mu, recon_logvar, inputs, z, mu, logvar)
        loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
        L1_error = L1_loss_function(inputs, recon_mu)
        L1_loss += L1_error.item()
        with open("err.txt", 'w') as file:
            file.write(str(6))
        if loss_ == 'L1':
            loss = L1_error + 1 / 1000 * KLD
        elif loss_ == 'ELBO':
            loss = BCE + KLD * min(epoch / 100.0, 1)
        elif loss_ == 'mixed_loss':
            loss = BCE * (1 - epoch / 1000) + L1_error * (epoch / 1000) + KLD * min(epoch / 100.0, 1)
        assert not np.isnan(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value += loss.item()
    loss_avg = loss_value / (len(train_data) // batch_size)
    info_loss = "Train epoch {}, L1 loss {}, recon {}, reg {}".format(epoch,
                                                                      L1_loss / (len(train_data) // batch_size),
                                                                      loss_bce / (len(train_data) // batch_size),
                                                                      loss_kld / (len(train_data) // batch_size))
    train_performances.append(info_loss)
    print(info_loss)
    with open("err.txt", 'w') as file:
        file.write(str(7))
    # for param_group in optimizer.param_groups:
    #  param_group['lr'] = initial_learning_rate*np.exp(-int(epoch/learning_factor))

    model.eval()
    loss_value = 0.0
    loss_bce = 0.0
    loss_kld = 0.0
    L1_loss = 0

    # Evaluate
    for j in range(len(val_data) // batch_size):
        inputs, laplacian, Di, DiA = sample_batch(val_data)
        if  operator == "lap_old":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L)
        if  operator == "lap_adj":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L_adj, mean_shape, L_adj)
        if operator == "lap" or operator == "lap_norm":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L, mean_shape, mean_L)
        if operator == "dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA, mean_shape, mean_Di, mean_DiA)
        if operator == "simple_dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, Di.transpose(1, 0), mean_shape, mean_simple_Di,
                                                          mean_simple_Di.transpose(1, 0))
        BCE, KLD = loss_function(recon_mu, recon_logvar, inputs, z, mu, logvar)
        loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
        L1_error = L1_loss_function(inputs, recon_mu)
        L1_loss += L1_error.item()

    loss_avg = loss_value / (len(val_data) // batch_size)
    info_loss = "Val epoch {}, L1 loss {}, recon {}, reg {}".format(epoch,
                                                                    L1_loss / (len(val_data) // batch_size),
                                                                    loss_bce / (len(val_data) // batch_size),
                                                                    loss_kld / (len(val_data) // batch_size))
    val_performances.append(info_loss)
    print(info_loss)
    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        torch.save(model.state_dict(), "{}/model_epoch{}.pt".format(operator + '_' + version, epoch))
        torch.save(optimizer.state_dict(), "{}/optimizer_epoch{}.pt".format(operator + '_' + version, epoch))
        with open("{}/train_performances.txt".format(operator + '_' + version), 'w') as file:
            file.write(str(train_performances))
        with open("{}/val_performances.txt".format(operator + '_' + version), 'w') as file:
            file.write(str(val_performances))
        minioClient.fput_object('coma', "{}/model_epoch{}.pt".format(operator + '_' + version, epoch),
                                "{}/model_epoch{}.pt".format(operator + '_' + version, epoch))
        minioClient.fput_object('coma', "{}/optimizer_epoch{}.pt".format(operator + '_' + version, epoch),
                                "{}/optimizer_epoch{}.pt".format(operator + '_' + version, epoch))
        minioClient.fput_object('coma', "{}/train_performances.txt".format(operator + '_' + version),
                                "{}/train_performances.txt".format(operator + '_' + version))
        minioClient.fput_object('coma', "{}/val_performances.txt".format(operator + '_' + version),
                                "{}/val_performances.txt".format(operator + '_' + version))

torch.cuda.empty_cache()

import gc

gc.collect()
# @title test
num_evaluation = 500
L1_error = np.zeros((num_evaluation))
euclidean_error = np.zeros((num_evaluation))
euclidean_dist = np.zeros((num_evaluation))

mus = []
label_mus = []
for i in range(num_evaluation):
    sampling = np.random.choice(len(test_data), batch_size)
    batch = []
    label_batch = []
    for s in sampling:
        batch.append(test_data[s])
        label_batch.append(test_labels[s])

    inputs, laplacian, Di, DiA = sample_batch(batch, False)
    if operator == "lap_old":
        recon_mu, recon_logvar, z, mu, logvar = model(inputs, L)
    if operator == "lap_adj":
        recon_mu, recon_logvar, z, mu, logvar = model(inputs, L_adj, mean_shape, L_adj)
    if operator == "lap" or operator == "lap_norm":
        recon_mu, recon_logvar, z, mu, logvar = model(inputs, L, mean_shape, mean_L)
    if operator == "dirac":
        recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA, mean_shape, mean_Di, mean_DiA)
    if operator == "simple_dirac":
        recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, Di.transpose(1, 0), mean_shape, mean_simple_Di,
                                                      mean_simple_Di.transpose(1, 0))

    mus.append(mu.detach().cpu())
    label_mus.append(label_batch)

    euclidean_error[i], euclidean_dist[i] = euclidean(inputs, recon_mu)
    L1_error[i] = L1_loss_function(inputs, recon_mu).item()
print("Euclidean Mean Error= ", euclidean_dist.mean())
print("Euclidean std Error= ", euclidean_dist.std())
print("Euclidean Mean %Error= ", euclidean_error.mean())
print("Euclidean std %Error= ", euclidean_error.var())

np.save('{}/mus.npy'.format(operator + '_' + version), [mu.detach().cpu().numpy() for mu in mus])
np.save('{}/label_mus.npy'.format(operator + '_' + version), label_mus)
np.save('{}/recon_samples.npy'.format(operator + '_' + version), recon_mu.detach().cpu())
np.save('{}/input_samples.npy'.format(operator + '_' + version), inputs.detach().cpu())