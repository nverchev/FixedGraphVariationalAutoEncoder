







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
