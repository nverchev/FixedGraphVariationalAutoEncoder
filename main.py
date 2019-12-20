from models import LapVAE, DirVAE
from util import sparse_diag_cat,process
from losses import *
import torch
import numpy as np
import glob
import numpy as np
import re
import os
from torch import optim

#Change this
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
parser.add_argument('--initial-learning_rate', type=float, default=0.001, metavar='N',
                    help='num of training epochs (default: 0.001)')
parser.add_argument('--model', default="lap",
                    help='lap | dirac | simple_dirac')
parser.add_argument('--version', default="hpc_temp")
parser.add_argument('--load-version', type=int, default=1000, metavar='N',
                    help="-1 don't load,0 most recent otherwise epoch")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

operator = args.model
num_epoch = args.num_epoch
batch_size= args.batch_size
num_features=args.num_features
dim_latent=args.dim_latent
num_blocks_encoder= args.num_blocks_encoder
num_blocks_decoder=args.num_blocks_decoder
initial_learning_rate=args.initial_learning_rate
weight_decay=args.weight_decay
learning_factor=args.adaptive_cycle
version=args.version
load=args.load_version
def load_sparse(sample):
    if operator == "lap": 
        sample['L']=torch.sparse.FloatTensor(sample['L.ind'], sample['L.val'], sample['L.size'])
    if operator == "dirac":
        sample['Di']=torch.sparse.FloatTensor(sample['Di.ind'], sample['Di.val'], sample['Di.size'])
        sample['DiA']=torch.sparse.FloatTensor(sample['DiA.ind'], sample['DiA.val'], sample['DiA.size'])
    if operator == "simple_dirac":
        sample['Di'] = torch.sparse.FloatTensor(sample['Di.ind'], sample['Di.val'], sample['Di.size'])
    del sample['Di.ind'], sample['Di.val'], sample['Di.size']
    try:
        del sample['L.ind'], sample['L.val'], sample['L.size']
        del sample['DiA.ind'], sample['DiA.val'], sample['DiA.size']
    except:
        pass
    return sample


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ErrorFile=[]
with open("errors.txt", 'w') as file:
    file.write(str(ErrorFile))

data=np.array([])
path_list=[]
for path in sorted(glob.glob("../data_vo/preproc_data/*")):
    for sample in torch.load(path):
        skip=0
        for tensor in [key for key in sample.keys() if not re.search("size", key)]:
            if torch.isnan(sample[tensor]).any():
                skip=1
        if not skip:
            data=np.hstack([data,load_sparse(sample)])
            path_list.append(path)

with open("errors.txt", 'w') as file:
    file.write(str(ErrorFile))
for path in sorted(glob.glob("../scratch_kyukon_vo/preproc_data/*")):
    for sample in torch.load(path):
        skip=0
        for tensor in [key for key in sample.keys() if not re.search("size", key)]:
            if torch.isnan(sample[tensor]).any():
                skip=1
        if not skip:
            data=np.hstack([data,load_sparse(sample)])
            path_list.append(path)

with open("errors.txt", 'w') as file:
    file.write(str(ErrorFile))
for path in sorted(glob.glob("../scratch_phanpy_vo/preproc_data/*")):
    for sample in torch.load(path):
        skip=0
        for tensor in [key for key in sample.keys() if not re.search("size", key)]:
            if torch.isnan(sample[tensor]).any():
                skip=1
        if not skip:
            data=np.hstack([data,load_sparse(sample)])
            path_list.append(path)
with open("errors.txt", 'w') as file:
    file.write(str(ErrorFile))
test_labels=[]
for i,file in enumerate(path_list):
    if i%10==9:
        test_labels.append(int(file[-8:-6]))


train_data=[]
val_data=[]
test_data=[]
for i,sample in enumerate(data):
    if i%10==8:
        val_data.append(sample)
    elif i%10==9:
        test_data.append(sample)
    else:
        train_data.append(sample)



def sample_batch(samples,train=True):
    indices = []
    if train:
        for b in range(batch_size):
          ind = np.random.randint(0, len(samples))
          indices.append(ind)
    else:
        indices=list(range(len(samples)))

    inputs = torch.zeros(batch_size, num_vertices, 3)
    faces = torch.zeros(batch_size, num_faces, 3).long()
    laplacian = []
    Di = []
    DiA = []
    for b, ind in enumerate(indices):
        inputs[b] = samples[ind]['V']
        laplacian.append(samples[ind]['L'])
        if operator == "dirac":
            Di.append(samples[ind]['Di'])
            DiA.append(samples[ind]['DiA'])
        if operator == "simple_dirac":
            Di.append(samples[ind]['Di'])
    if operator=="lap":
        laplacian = sparse_diag_cat(laplacian,num_vertices,num_vertices)
        return inputs.to(device), laplacian.to(device),None,None

    if operator== "dirac":
        Di = sparse_diag_cat(Di, 4 *num_faces, 4 * num_vertices)
        DiA = sparse_diag_cat(DiA, 4 * num_vertices, 4 * num_faces)
        return inputs.to(device), None, Di.to(device), DiA.to(device)
    if operator== "simple_dirac":
        Di = sparse_diag_cat(Di, 4 *num_faces, 4 * num_vertices)
        return inputs.to(device), None, Di.to(device), None

mean_shape = torch.load('mean_shape.pt').to(device)
if operator == "lap":
    mean_L_v = torch.load("mean_L_v.pt")
    mean_L_i = torch.load("mean_L_i.pt")
    mean_L_s = torch.load("mean_L_s.pt")
    mean_L = torch.sparse.FloatTensor(mean_L_i, mean_L_v, mean_L_s).cpu()
    mean_L = sparse_diag_cat([mean_L for _ in range(batch_size)], num_vertices, num_vertices).to(device).detach()

elif operator == "dirac":
    mean_Di_v = torch.load("mean_simple_Di_v.pt")
    mean_Di_i = torch.load("mean_simple_Di_i.pt")
    mean_Di_s = torch.load("mean_simple_Di_s.pt")
    mean_DiA_v = torch.load("mean_simple_DiA_v.pt")
    mean_DiA_i = torch.load("mean_simple_DiA_i.pt")
    mean_DiA_s = torch.load("mean_simple_DiA_s.pt")
    mean_Di = torch.sparse.FloatTensor(mean_Di_i, mean_Di_v, mean_Di_s).cpu()
    mean_Di = sparse_diag_cat([mean_Di for _ in range(batch_size)], 4 * num_faces, 4 * num_vertices).to(
        device).detach()
    mean_DiA = torch.sparse.FloatTensor(mean_DiA_i, mean_DiA_v, mean_DiA_s).cpu()
    mean_DiA = sparse_diag_cat([mean_DiA for _ in range(batch_size)], 4 * num_faces, 4 * num_vertices).to(
        device).detach()

elif operator == "simple_dirac":
    mean_simple_Di_v = torch.load("mean_simple_Di_v.pt")
    mean_simple_Di_i = torch.load("mean_simple_Di_i.pt")
    mean_simple_Di_s = torch.load("mean_simple_Di_s.pt")
    mean_simple_Di = torch.sparse.FloatTensor(mean_simple_Di_i, mean_simple_Di_v, mean_simple_Di_s).cpu()
    mean_simple_Di = sparse_diag_cat([mean_simple_Di for _ in range(batch_size)], 4 * num_faces, 4 * num_vertices).to(
        device).detach()

if operator == "lap":
    model = LapVAE(num_features,num_blocks_encoder,num_blocks_decoder,dim_latent)
else:
    model = DirVAE(num_features,num_blocks_encoder,num_blocks_decoder,dim_latent)

init_epoch=1
train_performances=[]
val_performances=[]
optimizer = optim.Adam(model.parameters(), initial_learning_rate, weight_decay=weight_decay)
ErrorFile.append(3)
with open("errors.txt", 'w') as file:
    file.write(str(ErrorFile))
try:
    os.mkdir(operator+'_'+version)
except:
    pass
if load+1:
    try:

        for file in minioClient.list_objects('coma',recursive=True):
            file= file.object_name
            if file.split("/")[0]==operator+'_'+version:
                minioClient.fget_object('coma',file,file)
        list_models={int(re.sub("\D", "",file)) :file for file in glob.glob('{}/model_epoch*'.format(operator+'_'+version))}
        list_optimizer={int(re.sub("\D", "",file)) :file for file in glob.glob('{}/optimizer_epoch*'.format(operator+'_'+version))}
        max_epoch=load if load else max(list_models.keys())
        init_epoch=max_epoch+1
        model.load_state_dict(torch.load(list_models[max_epoch]))
        model.to(device)
        optimizer.load_state_dict(torch.load(list_optimizer[max_epoch]))
        with open("{}/train_performances.txt".format(operator+'_'+version),'r') as file:
            train_performances=eval(file.read())
        with open("{}/val_performances.txt".format(operator+'_'+version),'r') as file:
            val_performances=eval(file.read())
        print("loaded: ",list_models[max_epoch])
    except:
        print("No saved models!")

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))
model.cuda()
#fixed_noise_ = torch.FloatTensor(batch_size, 1, 100).normal_(0, 1)
#fixed_noise_ = fixed_noise_.cuda()

for epoch in range(init_epoch,init_epoch+num_epoch):
    model.train()
    loss_value = 0.0
    loss_bce = 0.0
    loss_kld = 0.0
    L1_loss=0
    # Train
    for j in range(len(train_data) // batch_size):
        inputs, L, Di, DiA= sample_batch(train_data)
        if operator == "lap":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L, mean_shape, mean_L)
        if operator == "dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA, mean_shape, mean_Di, mean_DiA)
        if operator == "simple_dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, Di.transpose(1, 0), mean_shape, mean_simple_Di,mean_simple_Di.transpose(1, 0))
        BCE, KLD = loss_function(recon_mu, recon_logvar,  inputs, z, mu, logvar)
        loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
        L1_error=L1_loss_function(inputs,recon_mu)
        L1_loss += L1_error.item()
        loss = L1_error+ 1/1000*KLD# * min(epoch/100.0, 1)
        #loss = BCE*(1-epoch/1000) +L1_error*(epoch/1000)+ KLD * min(epoch/100.0, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value += loss.item()
    loss_avg=loss_value / (len(train_data) // batch_size)
    info_loss="Train epoch {}, L1 loss {}, recon {}, reg {}".format(epoch,
                  L1_loss/ (len(train_data) // batch_size),
                  loss_bce / (len(train_data) // batch_size),
                  loss_kld / (len(train_data) // batch_size))
    train_performances.append(info_loss)

    # for param_group in optimizer.param_groups:
  #  param_group['lr'] = initial_learning_rate*np.exp(-int(epoch/learning_factor))

    model.eval()
    loss_value = 0.0
    loss_bce = 0.0
    loss_kld = 0.0
    L1_loss=0

    # Evaluate
    for j in range(len(val_data) // batch_size):
        inputs, laplacian, Di, DiA= sample_batch(val_data)
        if operator == "lap":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, L, mean_shape, mean_L)
        if operator == "dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA, mean_shape, mean_Di, mean_DiA)
        if operator == "simple_dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, Di.transpose(1, 0), mean_shape, mean_simple_Di,mean_simple_Di.transpose(1, 0))
        BCE, KLD = loss_function(recon_mu, recon_logvar,  inputs, z, mu, logvar)
        loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
        L1_error=L1_loss_function(inputs,recon_mu)
        L1_loss += L1_error.item()
        #loss = BCE*(1-epoch/1000) +L1_error*(epoch/1000)+ KLD * min(epoch/100.0, 1)
        loss = L1_error+ 1/1000*KLD# * min(epoch/100.0, 1)

    loss_avg=loss_value / (len(val_data) // batch_size)
    info_loss="Val epoch {}, L1 loss {}, recon {}, reg {}".format(epoch,
                  L1_loss/ (len(val_data) // batch_size),
                  loss_bce / (len(val_data) // batch_size),
                  loss_kld / (len(val_data) // batch_size))
    val_performances.append(info_loss)

    if(epoch%50==0):
        model.eval()
        torch.save(model.state_dict(), "{}/model_epoch{}.pt".format(operator+'_'+version,epoch))
        torch.save(optimizer.state_dict(), "{}/optimizer_epoch{}.pt".format(operator+'_'+version,epoch))
        with open("{}/train_performances.txt".format(operator+'_'+version),'w') as file:
            file.write(str(train_performances))
        with open("{}/val_performances.txt".format(operator+'_'+version),'w') as file:
            file.write(str(val_performances))
        minioClient.fput_object('coma', "{}/model_epoch{}.pt".format(operator+'_'+version,epoch), "{}/model_epoch{}.pt".format(operator+'_'+version,epoch))
        minioClient.fput_object('coma', "{}/optimizer_epoch{}.pt".format(operator+'_'+version,epoch), "{}/optimizer_epoch{}.pt".format(operator+'_'+version,epoch))
        minioClient.fput_object('coma', "{}/train_performances.txt".format(operator+'_'+version), "{}/train_performances.txt".format(operator+'_'+version))
        minioClient.fput_object('coma', "{}/val_performances.txt".format(operator+'_'+version), "{}/val_performances.txt".format(operator+'_'+version))

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
    sampling = np.random.choice(len(test_data), 32)
    batch = []
    label_batch = []
    for s in sampling:
        batch.append(test_data[s])
        label_batch.append(test_labels[s])

    inputs, laplacian, Di, DiA = sample_batch(batch, False)
    if operator == "lap":
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
print("Euclidean std Error= ", euclidean_dist.std()())
print("Euclidean Mean %Error= ", euclidean_error.mean())
print("Euclidean std %Error= ", euclidean_error.var())

np.save('{}/mus.npy'.format(operator+'_'+version),[mu.detach().cpu().numpy() for mu in mus])
np.save('{}/label_mus.npy'.format(operator+'_'+version),label_mus)
np.save('{}/recon_samples.npy'.format(operator+'_'+version),recon_mu.detach().cpu())
np.save('{}/input_samples.npy'.format(operator+'_'+version),inputs.detach().cpu())


