from tarfile import XHDTYPE
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model
from protonets.models.encoder.default import C64
from protonets.models.encoder.GoogleKWS import cnn_trad_fpool3
from protonets.models.encoder.TCResNet import TCResNet8, TCResNet8Dilated
from protonets.models.encoder.CNN_BiLSTM import CNN_BiLSTM
from protonets.models.encoder.MLP import MLP
from protonets.models.encoder.MLP import MyModel
#from torch.utils.tensorboard import SummaryWriter
import os
from protonets.models.encoder.ResNet18_Attention import ResNet18
from protonets.models.encoder.ResNet18_Attention_kernel_v2 import ResNet18_kernel_v2
from protonets.models.encoder.ResNet18_Attention_dilation_v2 import ResNet18_dilation_v2
from protonets.models.encoder.ResNet18_Attention_kernel_dilation_v2 import ResNet18_kernel_dilation_v2
from .utils import euclidean_dist

class Protonet(nn.Module):
    def __init__(self, encoder, encoding):
        super(Protonet, self).__init__()
        self.encoding = encoding
        self.encoder = encoder
        #self.write = False

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        #if not self.write:
            #writer = SummaryWriter('runs/{}'.format(self.encoding))
            #writer.add_graph(self.encoder, x)
            #writer.close()
            #self.write = True
        # x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        # if self.encoding == "Triplet":
          # emb_space = load_triplet_model()
          # with torch.no_grad():
          #   # print("xs: ", x.shape)
          #   spec = x # torch.Size([28, 1, 16, 40])
          #   # print("xss: ",spec[1].shape)
          #   # spec = spec[1].reshape(x[1].shape[0], x[1].shape[2], x[1].shape[1]) #(1, 40, 16)
          #   # print("xss: ",spec.shape)
          #   # spec = spec.reshape(spec.shape[0], spec.shape[3], spec.shape[2]) #(batch_size, 40, 16)
          #   # x_emb = emb_space.spec_to_embedding(spec).detach().cpu().mean(dim=0).cuda()
          #   x_emb = []
          #   for i in x:
          #     # print("shape x[i]: ", i.shape, type(x), type(i))
          #     i = i.reshape(i.shape[0], i.shape[2], i.shape[1])
          #     x_emb_i = emb_space.spec_to_embedding(i).detach().cpu().mean(dim=0).cuda()
          #     x_emb.append(x_emb_i)
          # x_emb = n.array(x_emb)
          # # print(x_emb.shape)
        # print("x shape: ", x.shape) # (28, 1, 16)
        if self.encoding == "Triplet":
          z = self.encoder.forward(x) #(28, 1, 48)
          z = z.reshape(z.shape[0], z.shape[2])
          # print(z.shape)
          # z_dim = z.size(-1)
          # log_p_y = F.log_softmax(z, dim=1).view(n_class, n_query, -1)
          # pass
        
        else:
          z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        # #START: use distances to predict
        # separated_dists= dists.view(n_class, n_query, -1)
        # loss_val = separated_dists.gather(2, target_inds).squeeze().view(-1).mean()
        # _, y_hat = separated_dists.min(2)
        # # print('target_inds: ', target_inds)
        # #END: use distances to predict

        #START: use log_softmax
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        # print('log_p_y:', log_p_y)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        #END: use log_softmaX
        
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        # print("target_inds ", target_inds.squeeze())
        # print("y_hat", y_hat)

        # #start: print class and prediction
        # first = 0
        # second = 0
        # for i in range(target_inds.squeeze()[0].shape[0]): #nhãn 0
        #   if target_inds.squeeze()[0][i] == y_hat[0][i]:
        #     first += 1
        # for j in range(target_inds.squeeze()[1].shape[0]): #nhãn 1
        #   if target_inds.squeeze()[1][j] == y_hat[1][j]:
        #     second += 1
        # first_len = target_inds.squeeze()[0].shape[0]
        # second_len = target_inds.squeeze()[1].shape[0]
        # f_str = str(first) + "/" + str(first_len)
        # s_str = str(second) + "/" + str(second_len)
        # print("\nPREDICT: ", f_str, s_str) 
        # #end: print class and prediction
        
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


def get_enocder(encoding, x_dim, hid_dim, out_dim):
    if encoding == 'C64':
        return C64(x_dim[0], hid_dim, out_dim)
    elif encoding == 'cnn-trad-fpool3':
        return cnn_trad_fpool3(x_dim[0], hid_dim, out_dim)
    elif encoding == 'TCResNet8':
        return TCResNet8(x_dim[0], x_dim[1], x_dim[2])
    elif encoding == 'TCResNet8Dilated':
        return TCResNet8Dilated(x_dim[0], x_dim[1], x_dim[2])
    elif encoding == 'CNN_BiLSTM':
        # return CNN_BiLSTM()
        # return TCResNet8Dilated(x_dim[0], x_dim[1], x_dim[2])
        return CNN_BiLSTM(x_dim[0], x_dim[1], x_dim[2])
    elif encoding == 'Triplet':
        # return CNN_BiLSTM()
        # print("x_dim[0]: ", x_dim[0])
        return MLP(16, 64, 48)
    elif encoding == 'Attention':
        return ResNet18()
    elif encoding == 'Attention_kernel_v2':
        return ResNet18_kernel_v2()
    elif encoding == 'Attention_dilation_v2':
        return ResNet18_dilation_v2()
    elif encoding == 'Attention_kernel_dilation_v2':
        return ResNet18_kernel_dilation_v2()

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    encoding = kwargs['encoding']
    encoder = get_enocder(encoding, x_dim, hid_dim, z_dim)
    return Protonet(encoder, encoding)

def load_triplet_model():
  # load checkpoint
  DATA_PATH = '/content/drive/MyDrive/cough_detection/fewshot_fed_triplet'
  DEVICE = 'cpu'
  S = torch.load(os.path.join(DATA_PATH, 'checkpoints/last.ckpt'), map_location=torch.device(DEVICE))['state_dict']
  NS = {k[6:]: S[k] for k in S.keys() if (k[:5] == 'model')}

  # load model
  model = MyModel()
  model.load_state_dict(NS)
  model = model.eval()
  return model
