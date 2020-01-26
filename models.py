import torch
import torch.nn as nn
import torch.nn.functional as F
num_vertices = 5023
num_faces = 9976

class GraphConv1x1(nn.Module):
    def __init__(self, num_inputs, num_outputs, batch_norm=None):
        super(GraphConv1x1, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.batch_norm = batch_norm
        if self.batch_norm == "pre":
            self.bn = nn.BatchNorm1d(num_inputs, track_running_stats=False)
        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs, track_running_stats=False)
        self.fc = nn.Linear(num_inputs, num_outputs, bias=True)

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
        xs = [x, torch.mm(L, x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)
        x = F.elu(x)
        xs = [x, torch.mm(L, x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)

        x = self.bn_fc1(x)
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
        x = x.view(batch_size * num_faces * 4, num_inputs // 4)
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
            self.bn = nn.BatchNorm1d(num_inputs, track_running_stats=False)
        if self.batch_norm == "post":
            self.bn = nn.BatchNorm1d(num_outputs, track_running_stats=False)
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
        xs = [x, torch.mm(L, x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)
        x = self.bn_fc0(x)
        x = F.elu(x)
        xs = [x, torch.mm(L, x.view(-1, feat)).view(batch, node, feat)]
        x = torch.cat(xs, 2)
        x = self.bn_fc1(x)
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
        x = x.view(batch_size * num_faces * 4, num_inputs // 4)
        x = torch.mm(DiA, x)
        x = x.view(batch_size, num_nodes, num_inputs)
        x = torch.cat([x_in, x], 2)
        x = self.bn_fc1(x)
        v_out = x
        return v + v_out, f_out


class LapEncoder_old(nn.Module):
    def __init__(self,num_features,num_blocks_encoder,dim_latent):
        super().__init__()

        self.conv1 = GraphConv1x1(3, num_features, batch_norm=None)

        self.num_layers = num_blocks_encoder
        for i in range(self.num_layers):
            module = LapResNet(num_features)
            self.add_module("rn{}".format(i), module)

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

        return self.fc_mu(x), self.fc_logvar(x)


class LapDecoder_old(nn.Module):
    def __init__(self,num_features,num_blocks_decoder,dim_latent):
        super().__init__()

        self.conv_inputs = GraphConv1x1(dim_latent, num_features, batch_norm=None)
        self.num_layers = num_blocks_decoder
        for i in range(self.num_layers):
            module = LapResNet(num_features)
            self.add_module("rn{}".format(i), module)

        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")

        self.fc_mu = GraphConv1x1(num_features, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, L):
        batch_size, num_nodes, _ = inputs.size()
        x = self.conv_inputs(inputs)

        for i in range(self.num_layers):
            x = self._modules['rn{}'.format(i)](L, x)

        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)

        mu = self.fc_mu(x)

        y = self.fc_logvar.expand_as(mu).contiguous()

        return mu + inputs, y
class LapVAE_old(nn.Module):

    def __init__(self,num_features,num_blocks_encoder,num_blocks_decoder,dim_latent):
        super().__init__()

        self.encoder = LapEncoder_old(num_features,num_blocks_encoder,dim_latent)
        self.decoder = LapDecoder_old(num_features,num_blocks_decoder,dim_latent)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x,  L):
        mu, logvar = self.encoder(x, L)

        z = self.reparametrize(mu, logvar)

        z_ = z.unsqueeze(1)
        z_ = z_.repeat(3, flat_x.size(1), 1)

        recog_mu, recog_logvar = self.decoder(z_, L)
        return recog_mu, recog_logvar, z, mu, logvar

class LapEncoder(nn.Module):
    def __init__(self,num_features,num_blocks_encoder,dim_latent):
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
    def __init__(self,num_features,num_blocks_decoder,dim_latent):
        super().__init__()
        self.conv_shape = GraphConv1x1(3, num_features, batch_norm=None)
        self.conv_latent = GraphConv1x1(num_features, num_features, batch_norm=None)
        self.num_layers = num_blocks_decoder
        for i in range(self.num_layers):
            module = LapResNet(num_features)
            self.add_module("rn{}".format(i), module)
        self.dense_latent1 = nn.Linear(dim_latent, num_features)
        self.dense_latent2 = nn.Linear(num_features, num_features)
        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")
        self.fc_mu = GraphConv1x1(num_features, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, L, mean_shape):
        x = self.conv_shape(mean_shape.unsqueeze(0).repeat(inputs.size(0), 1, 1))
        x = F.elu(x)
        l = self.dense_latent1(inputs)
        l = self.dense_latent2(l)
        x = torch.mul(x, l.unsqueeze(1))
        x = self.conv_latent(x)
        for i in range(self.num_layers):
            x = self._modules['rn{}'.format(i)](L, x)
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)
        mu = self.fc_mu(x)
        y = self.fc_logvar.expand_as(mu).contiguous()

        return mu, y


class LapVAE(nn.Module):

    def __init__(self,num_features,num_blocks_encoder,num_blocks_decoder,dim_latent):
        super().__init__()
        self.encoder = LapEncoder(num_features,num_blocks_encoder,dim_latent)
        self.decoder = LapDecoder(num_features,num_blocks_decoder,dim_latent)

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
        recog_mu, recog_logvar = self.decoder(z, mean_L, mean_shape)
        return recog_mu, recog_logvar, z, mu, logvar


class DirEncoder(nn.Module):
    def __init__(self,num_features,num_blocks_encoder,dim_latent):
        super().__init__()

        self.conv1 = GraphConv1x1(3, num_features, batch_norm=None)

        self.num_layers = num_blocks_encoder
        for i in range(self.num_layers):
            module = DirResNet(num_features)
            self.add_module("rn{}".format(i), module)
        self.fc_hidden = nn.Linear(num_features, num_features)
        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")
        self.num_features=num_features
        self.fc_mu = nn.Linear(num_features, dim_latent)
        self.fc_logvar = nn.Linear(num_features, dim_latent)

    def forward(self, inputs, Di, DiA):
        batch_size, num_nodes, _ = inputs.size()
        v = self.conv1(inputs)
        f = torch.zeros(batch_size, num_faces, self.num_features)
        if v.is_cuda:
            f = f.cuda()
        for i in range(self.num_layers):
            v, f = self._modules['rn{}'.format(i)](Di, DiA, v, f)
        x = v
        x = F.elu(x)
        x = self.bn_conv2(x)
        x = F.elu(x)
        x = global_average(x).squeeze()
        x = self.fc_hidden(x)
        return self.fc_mu(x), self.fc_logvar(x)


class DirDecoder(nn.Module):
    def __init__(self,num_features,num_blocks_decoder,dim_latent):
        super().__init__()
        self.num_layers = num_blocks_decoder
        self.bn_conv2 = GraphConv1x1(num_features, num_features, batch_norm="pre")
        self.num_features=num_features
        self.fc_mu = GraphConv1x1(num_features, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))
        self.conv_shape = GraphConv1x1(3, num_features, batch_norm=None)
        self.conv_latent = GraphConv1x1(num_features, num_features, batch_norm=None)
        self.num_layers = num_blocks_decoder
        for i in range(self.num_layers):
            module = DirResNet(num_features)
            self.add_module("rn{}".format(i), module)
        self.dense_latent1 = nn.Linear(dim_latent, num_features)
        self.dense_latent2 = nn.Linear(num_features, num_features)
        self.fc_mu = GraphConv1x1(num_features, 3, batch_norm=None)
        self.fc_logvar = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, inputs, Di, DiA, mean_shape):
        x = self.conv_shape(mean_shape.unsqueeze(0).repeat(inputs.size(0), 1, 1))
        x = F.elu(x)
        l = self.dense_latent1(inputs)
        l = self.dense_latent2(l)
        x = torch.mul(x, l.unsqueeze(1))
        v = self.conv_latent(x)
        f = torch.zeros(x.size(0), num_faces, self.num_features).cuda()
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
    def __init__(self,num_features,num_blocks_encoder,num_blocks_decoder,dim_latent):
        super().__init__()

        self.encoder = DirEncoder(num_features,num_blocks_encoder,dim_latent)
        self.decoder = DirDecoder(num_features,num_blocks_decoder,dim_latent)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if mu.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, Di, DiA,mean_shape, mean_Di,mean_DiA):
        mu, logvar = self.encoder(x, Di, DiA)
        z = self.reparametrize(mu, logvar)
        recog_mu, recog_logvar = self.decoder(z, mean_Di, mean_DiA, mean_shape)
        return recog_mu, recog_logvar, z, mu, logvar
