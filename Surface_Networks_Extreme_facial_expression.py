
import torch
import numpy as np
import glob
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import os
import math
from torch import optim

#Change this
from minio import Minio
from minio.error import ResponseError
minioClient = Minio('s3nverchev.ugent.be',
                  access_key='M12AX6PRAW2HUPUHPTL0',
                  secret_key='yC2UlxBD+exlGz5S+zLaVclYUvRQ9D8msgOMVAWh',
                  secure=True)


operator = "lap" #@param [ "lap", "mlp", "avg", "dirac", "quatlap", "quatmlp", "quatavg", "quatdirac"]


num_epoch = 1000  #@param {type: "number"}
batch_size= 64 #@param {type: "number"}
num_features=64 #@param {type: "number"}
dim_latent=8 #@param {type: "number"}
num_blocks_encoder=2 #@param {type: "number"}
num_blocks_decoder=3 #@param {type: "number"}
initial_learning_rate=0.01 #@param {type: "number"}
weight_decay=0.000001 #@param {type: "number"}
learning_factor=100 #@param {type: "number"}



# In[5]:


#@title Defining the layers { display-mode: "form" }

#@markdown Here is the list:
#@markdown - 1X1 Quat_Convolution
#@markdown - Global average pooling
#@markdown - LapResNet
#@markdown - DirResNet
#@markdown - AvgResNet
#@markdown - MlpResNet
#@markdown - QuatLapResNet
#@markdown - QuatDirResNet
#@markdown - QuatAvgResNet
#@markdown - QuatMlpResNet


class GraphConv1x1(nn.Module):
    def __init__(self, num_inputs, num_outputs, batch_norm=None):
        super(GraphConv1x1, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm
        if self.batch_norm == "pre":
            self.bn = nn.BatchNorm1d(num_inputs,track_running_stats=False)
        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs,track_running_stats=False)
        self.fc = nn.Linear(num_inputs, num_outputs,bias=True)

    def forward(self, x):
        batch_size, num_nodes, num_inputs = x.size()
        assert num_inputs == self.num_inputs
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        if self.batch_norm == "pre":
            x = self.bn(x)
        x = self.fc(x)
        if self.batch_norm == "post":
            x = self.bn(x)
        x = x.view(batch_size, num_nodes, self.num_outputs)
        return x

class LapResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, inputs):
        x = inputs
        x = F.elu(x)
        batch, node, feat = x.size()
        xs = [x, torch.mm(L,x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)
        x = F.elu(x)
        xs = [x, torch.mm(L,x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)

        x = self.bn_fc1(x)
        return x + inputs
class AvgResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L,inputs):
        x = inputs
        x = F.elu(x)
        xs = [x, global_average(x).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)
        x = F.elu(x)
        xs = [x, global_average(x).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs

class MlpResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn0 = GraphBatchNorm(num_outputs)
        self.fc0 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None)
        self.bn1 = GraphBatchNorm(num_outputs)
        self.fc1 = GraphConv1x1(num_outputs, num_outputs, batch_norm=None)
    def forward(self, L, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.elu(x)
        x = self.fc0(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.fc1(x)
        return x + inputs

class DirResNet(nn.Module):
    def __init__(self, num_outputs, res_f=False):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn_fc0 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = GraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.res_f = res_f

    def forward(self, Di, DiA, v, f):
        batch_size, num_nodes, num_inputs = v.size()
        _, num_faces, _ = f.size()
        x_in, f_in = F.elu(v), F.elu(f)
        x = x_in
        x = x.view(batch_size * num_nodes * 4, num_inputs // 4)
        x = torch.mm(Di, x)
        x = x.view(batch_size, num_faces, num_inputs)
        x = torch.cat([f_in, x], 2)
        x = self.bn_fc0(x)
        f_out = x
        x = F.elu(x)
        x = x.view(batch_size* num_faces * 4, num_inputs // 4)
        x = torch.mm(DiA, x)
        x = x.view(batch_size, num_nodes, num_inputs)
        x = torch.cat([x_in, x], 2)
        x = self.bn_fc1(x)
        v_out = x
        return v + v_out, f_out


class QuatGraphConv1x1(nn.Module):
    def __init__(self, num_inputs, num_outputs, batch_norm=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm
        if self.batch_norm == "pre":
            self.bn = nn.BatchNorm1d(num_inputs,track_running_stats=False)
        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs,track_running_stats=False)
        self.fc = QuaternionLinear(num_inputs, num_outputs)

    def forward(self, x):
        batch_size, num_nodes, num_inputs = x.size()
        assert num_inputs == self.num_inputs
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        if self.batch_norm == "pre":
            x = self.bn(x)
        x = self.fc(x)
        if self.batch_norm == "post":
            x = self.bn(x)
        x = x.view(batch_size, num_nodes, self.num_outputs)
        return x
      

def global_average(x):
        return torch.mean(x, dim=1)

class QuatLapResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn_fc0 = QuatGraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = QuatGraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, inputs):
        x = inputs
        x = F.elu(x)
        batch, node, feat = x.size()
        xs = [x, torch.mm(L,x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)
        x = F.elu(x)
        xs = [x, torch.mm(L,x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)
        return x + inputs
class QuatAvgResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn_fc0 = QuatGraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = QuatGraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")

    def forward(self, L, inputs):
        x = inputs
        x = F.elu(x)
        xs = [x, global_average(x ).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)
        x = F.elu(x)
        xs = [x, global_average(x).expand_as(x).contiguous()]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)

        return x + inputs

class QuatMlpResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn0 = GraphBatchNorm(num_outputs)
        self.fc0 = QuatGraphConv1x1(num_outputs, num_outputs, batch_norm=None)
        self.bn1 = GraphBatchNorm(num_outputs)
        self.fc1 = QuatGraphConv1x1(num_outputs, num_outputs, batch_norm=None)
    def forward(self, L, inputs):
        x = inputs
        x = self.bn0(x)
        x = F.elu(x)
        x = self.fc0(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.fc1(x)
        return x + inputs

class QuatDirResNet(nn.Module):
    def __init__(self, num_outputs, res_f=False):
        super().__init__()
        self.num_outputs = num_outputs
        self.bn_fc0 = QuatGraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.bn_fc1 = QuatGraphConv1x1(2 * num_outputs, num_outputs, batch_norm="pre")
        self.res_f = res_f

    def forward(self, Di, DiA, v, f):
        batch_size, num_nodes, num_inputs = v.size()
        _, num_faces, _ = f.size()
        x_in, f_in = F.elu(v), F.elu(f)
        x = x_in
        x = x.view(batch_size * num_nodes * 4, num_inputs // 4)
        x = torch.mm(Di, x)
        x = x.view(batch_size, num_faces, num_inputs)
        x = torch.cat([f_in, x], 2)
        x = self.bn_fc0(x)
        f_out = x
        x = F.elu(x)
        x = x.view(batch_size* num_faces * 4, num_inputs // 4)
        x = torch.mm(DiA, x)
        x = x.view(batch_size, num_nodes, num_inputs)
        x = torch.cat([x_in, x], 2)
        x = self.bn_fc1(x)
        v_out = x
        return v + v_out, f_out


class LapEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GraphConv1x1(3, num_features, batch_norm=None)

        self.num_layers = num_blocks_encoder
        for i in range(self.num_layers):
            module = LapResNet(num_features)
            self.add_module("rn{}".format(i), module)
        self.fc_hidden = nn.Linear(num_features, num_features)

        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")
        self.fc_mu = nn.Linear(num_features, dim_latent)
        self.fc_logvar = nn.Linear(num_features, dim_latent)


    def forward(self, inputs, L):
        _, num_nodes, _ = inputs.size()
        x = self.conv1(inputs)
        for i in range(self.num_layers):
            x = self._modules['rn{}'.format(i)](L, x)
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)
        x = global_average(x).squeeze()
        x = self.fc_hidden(x)
        
        return self.fc_mu(x), self.fc_logvar(x)


class LapDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_shape = GraphConv1x1(3, num_features, batch_norm=None)
        self.conv_latent = GraphConv1x1(num_features, num_features, batch_norm=None)
        self.num_layers = num_blocks_decoder
        for i in range(self.num_layers):
            module = LapResNet(num_features)
            self.add_module("rn{}".format(i), module)
        self.dense_latent1= nn.Linear(dim_latent,num_features)
        self.dense_latent2= nn.Linear(num_features,num_features)
        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")
        self.fc_mu = GraphConv1x1(num_features, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, L, mean_shape):
        x = self.conv_shape(mean_shape.unsqueeze(0).repeat(batch_size, 1, 1))
        x = F.elu(x)
        l=self.dense_latent1(inputs)
        l=self.dense_latent2(l)
        x = torch.mul(x,l.unsqueeze(1))

        x = self.conv_latent(x)
        x = F.elu(x)

        for i in range(self.num_layers):
            x = self._modules['rn{}'.format(i)](L, x)
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)
        mu = self.fc_mu(x)
        y = self.fc_logvar.expand_as(mu).contiguous()

        return mu , y


class LapVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = LapEncoder()
        self.decoder = LapDecoder()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, L, mean_shape, mean_L):
        mu, logvar = self.encoder(x, L)
        z = self.reparametrize(mu, logvar)
        recog_mu, recog_logvar = self.decoder( z, mean_L, mean_shape)
        return recog_mu, recog_logvar, z, mu, logvar



class DirEncoder(nn.Module):
    def __init__(self):
        super(DirEncoder, self).__init__()

        self.conv1 = GraphConv1x1(3, num_features, batch_norm=None)

        self.num_layers = num_blocks_encoder
        for i in range(self.num_layers):
            module = DirResNet(num_features)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")

        self.fc_mu = nn.Linear(num_features, dim_latent)
        self.fc_logvar = nn.Linear(num_features, dim_latent)

    def forward(self, inputs, Di, DiA):
        batch_size, num_nodes, _ = inputs.size()

        v = self.conv1(inputs)

        f = torch.zeros(batch_size, num_faces, num_features)

        if v.is_cuda:
            f = f.cuda()

        for i in range(self.num_layers):
            v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)

        x = v
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        x = global_average(x).squeeze()

        return self.fc_mu(x), self.fc_logvar(x)


class DirDecoder(nn.Module):
    def __init__(self):
        super(DirDecoder, self).__init__()
        self.graph= nn.Linear(dim_latent,n_nodes)
        self.conv1 = GraphConv1x1(1, num_features, batch_norm=None)
        self.num_layers = num_blocks_decoder

        for i in range(self.num_layers):
            module = DirResNet(num_features)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")

        self.fc_mu = GraphConv1x1(num_features, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, Di, DiA):
        x= self.graph(inputs)
        x = F.elu(x)
        v = x.unsqueeze(dim=2)
        v=self.conv1(v)
        f = torch.zeros(batch_size, num_faces, num_features).cuda()
        for i in range(self.num_layers):
            v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
        x = v
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)
        mu = self.fc_mu(x)
        y = self.fc_logvar.expand_as(mu).contiguous()

        return mu, y

class DirVAE(nn.Module):

    def __init__(self):
        super(DirVAE, self).__init__()

        self.encoder = DirEncoder()
        self.decoder = DirDecoder()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, Di, DiA):
        mu, logvar = self.encoder(x, Di, DiA)

        z = self.reparametrize(mu, logvar)
        recog_mu, recog_logvar = self.decoder( z, Di, DiA)
        return recog_mu, recog_logvar, z, mu, logvar


# In[8]:


#@title Loader
def load_sparse(x):
    if operator == "lap": 
        sample['L']=torch.sparse.FloatTensor(sample['L.ind'], sample['L.val'], sample['L.size'])
    if operator == "dirac":
        sample['Di']=torch.sparse.FloatTensor(sample['Di.ind'], sample['Di.val'], sample['Di.size'])
        sample['DiA']=torch.sparse.FloatTensor(sample['DiA.ind'], sample['DiA.val'], sample['DiA.size'])
    del sample['L.ind'],sample['L.val'],sample['L.size']
    del sample['Di.ind'],sample['Di.val'],sample['Di.size']
    del sample['DiA.ind'],sample['DiA.val'],sample['DiA.size']
    return x
data=np.array([])
for path in sorted(glob.glob("../data_vo/preproc_data/*")):
    for sample in torch.load(path):
        skip=0
        for tensor in ['V', 'F', 'L.val', 'L.ind', 'Di.val', 'Di.ind', 'DiA.val',  'DiA.ind']:
            if torch.isnan(sample[tensor]).any():
                skip=1
        if not skip:
            data=np.hstack([data,load_sparse(sample)])
for path in sorted(glob.glob("../scratch_kyukon_vo/preproc_data/*")):
    for sample in torch.load(path):
        skip=0
        for tensor in ['V', 'F', 'L.val', 'L.ind', 'Di.val', 'Di.ind', 'DiA.val',  'DiA.ind']:
            if torch.isnan(sample[tensor]).any():
                skip=1
        if not skip:
            data=np.hstack([data,load_sparse(sample)])
for path in sorted(glob.glob("../scratch_phanpy_vo/preproc_data/*")):
    for sample in torch.load(path):
        skip=0
        for tensor in ['V', 'F', 'L.val', 'L.ind', 'Di.val', 'Di.ind', 'DiA.val',  'DiA.ind']:
            if torch.isnan(sample[tensor]).any():
                skip=1
        if not skip:
            data=np.hstack([data,load_sparse(sample)])


# In[9]:


#@title data splitting
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


# In[10]:


#@title sample batching
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def sample_batch(samples):
    indices = []

    for b in range(batch_size):
        ind = np.random.randint(0, len(samples))
        indices.append(ind)

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




    if operator== "dirac":
        Di = sparse_diag_cat(Di, 4 *num_faces, 4 * num_vertices)
        DiA = sparse_diag_cat(DiA, 4 * num_vertices, 4 * num_faces)
        return inputs.to(device), None, Di.to(device), DiA.to(device)
    else:
        laplacian = sparse_diag_cat(laplacian,num_vertices,num_vertices)
        return inputs.to(device), laplacian.to(device),None,None
        


# In[11]:


num_vertices = 5023
num_faces = 9976
mean_shape=torch.stack([sample['V'] for sample in train_data]).mean(axis=0).to(device)
mean_values_L=torch.stack([sample['L'].coalesce()._values() for sample in train_data]).mean(axis=0)
mean_L_list= [torch.sparse.FloatTensor(train_data[0]['L'].coalesce().indices(), mean_values_L, train_data[0]['L'].coalesce().size()) for _ in range(batch_size)]
mean_L=sparse_diag_cat(mean_L_list,num_vertices,num_vertices).to(device)


# In[ ]:


#@title train
num_vertices = 5023
num_faces = 9976
version='hpc' #@param {type:"string"}
if operator == "lap":
    model = LapVAE()
else:
    model = DirVAE()

Load = False #@param {type:"boolean"}
init_epoch=1
train_performances=[]
val_performances=[]
optimizer = optim.Adam(model.parameters(), initial_learning_rate, weight_decay=weight_decay)

try:
    os.mkdir(operator+'_'+version)
except:
    pass
if Load:
    try:

        for file in minioClient.list_objects('coma',recursive=True):
            file= file.object_name
            if file.split("/")[0]==operator+'_'+version:
                minioClient.fget_object('coma',file,file)
        list_models={int(re.sub("\D", "",file)) :file for file in glob.glob('{}/model_epoch*'.format(operator+'_'+version))}
        list_optimizer={int(re.sub("\D", "",file)) :file for file in glob.glob('{}/optimizer_epoch*'.format(operator+'_'+version))}
        load_epoch=0#@param {type:"integer"}
        max_epoch=load_epoch if load_epoch else max(list_models.keys())
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

def log_normal_diag(z, mu, logvar):
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp())

def loss_function(recon_mu, recon_logvar, x, z, mu, logvar):
    x = x.view(x.size(0), -1)
    recon_mu = recon_mu.view(x.size(0), -1)
    recon_logvar = recon_logvar.view(x.size(0), -1)
    BCE = -(log_normal_diag(x, recon_mu, recon_logvar)).sum(1).mean()

    log_q = log_normal_diag(z, mu, logvar)
    log_p = log_normal_diag(z, z * 0, z * 0)

    KLD_element = log_q - log_p
    KLD = KLD_element.sum(1).mean()
    return BCE, KLD

def L1_loss_function(recon_mu,x):
    x = x.view(x.size(0), -1)
    recon_mu = recon_mu.view(x.size(0), -1)     
    return torch.abs(recon_mu-x).sum(1).mean()
    #return torch.abs(recon_mu-x).mean()

for epoch in range(init_epoch,init_epoch+num_epoch):
    model.train()
    loss_value = 0.0
    loss_bce = 0.0
    loss_kld = 0.0
    L1_loss=0
    # Train
    for j in range(len(train_data) // batch_size):
        inputs, laplacian, Di, DiA= sample_batch(train_data)
        if operator == "lap":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, laplacian,mean_shape,mean_L)
        if operator == "dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA)
        BCE, KLD = loss_function(recon_mu, recon_logvar,  inputs, z, mu, logvar)
        loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
        L1_error=L1_loss_function(inputs,recon_mu)
        L1_loss += L1_error.item()
        #loss = L1_error+ 1/1000*KLD * min(epoch/100.0, 1)
        loss = BCE*(1-epoch/1000) +L1_error*(epoch/1000)+ KLD * min(epoch/100.0, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value += loss.item()
    loss_avg=loss_value / (len(train_data) // batch_size)
    print("loss: {}".format(loss_avg))
    print("Train epoch {}, L1 loss {}, bce {}, kld {}".format(epoch,
          L1_loss / (len(train_data) // batch_size),
          loss_bce / (len(train_data) // batch_size),
          loss_kld / (len(train_data) // batch_size)))
    train_performances.append(loss_avg)        
  

   
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
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, laplacian,mean_shape,mean_L)
        if operator == "dirac":
            recon_mu, recon_logvar, z, mu, logvar = model(inputs, Di, DiA)
        BCE, KLD = loss_function(recon_mu, recon_logvar,  inputs, z, mu, logvar)
        loss_bce, loss_kld = loss_bce + BCE.item(), loss_kld + KLD.item()
        L1_error=L1_loss_function(inputs,recon_mu)
        L1_loss += L1_error.item()
        loss = BCE*(1-epoch/1000) +L1_error*(epoch/1000)+ KLD * min(epoch/100.0, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value += loss.item()


    loss_avg=loss_value / (len(val_data) // batch_size)
    val_performances.append(loss_avg)
    print("loss: ", loss_avg)        
    print("Val epoch {}, loss {}, bce {}, kld {}".format(epoch,
                  L1_loss/ (len(val_data) // batch_size),
                  loss_bce / (len(val_data) // batch_size),
                  loss_kld / (len(val_data) // batch_size)))
    if(epoch%100==0):
        model.eval()
        torch.save(model.state_dict(), "{}/model_epoch{}.pt".format(operator+'_'+version,epoch))
        torch.save(optimizer.state_dict(), "{}/optimizer_epoch{}.pt".format(operator+'_'+version,epoch))
        with open("{}/train_performances.txt".format(operator+'_'+version),'w') as file:
            file.write(str(train_performances))
        with open("{}/val_performances.txt".format(operator+'_'+version),'w') as file:
            file.write(str(val_performances))

        try:
            minioClient.fput_object('coma', "{}/model_epoch{}.pt".format(operator+'_'+version,epoch), "{}/model_epoch{}.pt".format(operator+'_'+version,epoch))
            minioClient.fput_object('coma', "{}/optimizer_epoch{}.pt".format(operator+'_'+version,epoch), "{}/optimizer_epoch{}.pt".format(operator+'_'+version,epoch))
            minioClient.fput_object('coma', "{}/train_performances.txt".format(operator+'_'+version), "{}/train_performances.txt".format(operator+'_'+version))
            minioClient.fput_object('coma', "{}/val_performances.txt".format(operator+'_'+version), "{}/val_performances.txt".format(operator+'_'+version))


        except ResponseError as err:
            print(err)

  # _, fixed_flat_inputs, fixed_mask, _, fixed_flat_laplacian, _, _, fixed_flat_Di, fixed_flat_DiA, fixed_faces = sample_batch(test_data)
  # fixed_noise = fixed_noise_.repeat(1, fixed_flat_inputs.size(1), 1)


  # if args.model == "lap":
  #     fake, _ = model.decoder(fixed_flat_inputs, fixed_noise, fixed_flat_laplacian, fixed_mask)
  # else:
  #     fake, _ = model.decoder(fixed_flat_inputs, fixed_noise, fixed_flat_Di, fixed_flat_DiA, fixed_mask)



# In[ ]:




