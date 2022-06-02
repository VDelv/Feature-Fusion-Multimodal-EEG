'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be
Copyright (C) 2021 - UMons
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''

import os
import math
import numpy as np

from tqdm import tqdm
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Function
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EEGDataset(Dataset):
    def __init__(self, label, eeg):
        self.sig = eeg
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        eeg = self.sig[idx]
        label = self.label[idx]

        return (eeg, label)

class EEGPhiDataset(Dataset):
    def __init__(self, label, eeg, phi):
        self.sig = eeg
        self.phi = phi
        self.label = label

    def __len__(self):
        return len(self.label)  

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        eeg = self.sig[idx]
        phi = self.phi[idx]
        label = self.label[idx]

        return (eeg, phi, label)

class Uni_PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(Uni_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class EEG_PositionalEncoding(nn.Module):
    def __init__(self, d_model, elec_pos, dropout=0.1, max_len=512):
        super(EEG_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        elec_pos = (elec_pos-elec_pos.min())/np.max(elec_pos-elec_pos.min())
        position = torch.tensor(np.sqrt((elec_pos**2).sum(1)))       
        position = (position-position.min())/torch.max(position - position.min()) + 0.1
        position *= 128

        position_ = torch.arange(0, max_len, dtype=torch.float)
        position_[0:position.shape[0]] = position
        position = position_.unsqueeze(1)

        pe = torch.zeros(max_len, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Encoder( nn.Module ):
    def __init__(self, e_dim, n_chan, f_dim = 20, dropout = 0.25, eeg=False, pos=None):
        super( Encoder, self ).__init__()
        self.e_dim = e_dim
        self.batch_norm = nn.BatchNorm1d(30)  
        self.n_chan = n_chan
        self.f_dim = f_dim #feature dimension
        self.dropout = dropout
        self.encoding = nn.Linear(self.f_dim, self.e_dim)
        if eeg:
            self.pos_encoding = EEG_PositionalEncoding(self.e_dim, pos, self.dropout) 
        else : 
            self.pos_encoding = Uni_PositionalEncoding(self.e_dim, self.dropout)
    def forward(self, x):
        x = torch.stack([self.encoding(x[:, i]) for i in range(self.n_chan)], 0) * math.sqrt(self.e_dim)  
        x = self.pos_encoding(x)
        return x

class Decoder( nn.Module ):
    def __init__(self, h_dim, n_chan, out_dim):
        super( Decoder, self ).__init__()
        self.h_dim = h_dim
        self.n_chan = n_chan
        self.decoding = nn.Linear(h_dim, out_dim)
    def forward(self, x):
        x = torch.stack([self.decoding(x[i]) for i in range(self.n_chan)], 0)
        x = x.transpose(0,1)
        return x
    
class Attention(nn.Module):
    def __init__(self, e_dim, h_dim, nhead, nlayer, n_chan, f_dim, dropout=0.25, eeg=False, pos=None, gpu=0):
        super( Attention, self ).__init__()
        if torch.cuda.is_available:
            self.encoder = Encoder(e_dim=e_dim, n_chan=n_chan, f_dim=f_dim, eeg=eeg, pos=pos).cuda(gpu)
        else : 
            self.encoder = Encoder(e_dim=e_dim, n_chan=n_chan, f_dim=f_dim, eeg=eeg, pos=pos)
        self.nhead = nhead
        self.nlayer = nlayer
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.dropout = dropout
        
        encoder_layer = TransformerEncoderLayer(d_model=self.e_dim, nhead=self.nhead, dim_feedforward=self.e_dim, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.nlayer)
        self.decoder = nn.Sequential(
            nn.Linear(n_chan*e_dim, e_dim),
            nn.ReLU(True),
        )
        
        self.s_max = nn.Softmax(dim=1)
        
        self.bn = nn.BatchNorm1d(e_dim)
        
    def forward(self, x):
        b_size = x.shape[0]
        x = self.encoder(x)
        x = self.transformer_encoder(x)
        
        ##v1
        #x = self.decoder(x.transpose(0,1).reshape(b_size,-1))
        #x = self.bn(x)
        
        #v2
        x = self.bn(x[-1])
        return x
    
'''
class MultiAttention(nn.Module):
    def __init__(self, spatial_dep, emb_dim, feat_dim=1, eeg=False, pos=None, n_class=1, gpu=0):
        super(MultiAttention, self ).__init__()
        self.spatial_attention = Attention(e_dim=256,h_dim=256, nhead=1, 
            nlayer=2, n_chan=spatial_dep, f_dim=emb_dim*feat_dim, eeg=eeg, pos=None)
        
        self.Classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(64, n_class),
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        b_size = x.shape[0]
        c_dim = x.shape[2]
        
        spatial_x = x.transpose(1,2).reshape(b_size, c_dim, -1)

        feat = self.spatial_attention(spatial_x)
        
        out = self.Classifier(feat)
        
        return out
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.25, n_head=2):
        super(MultiHeadAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_head = n_head
        
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_merge = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, v, k, q):
        b_size = q.shape[0]
        v = self.linear_v(v)
        k = self.linear_k(k)
        q = self.linear_q(q)
        
        att = self.attention(v, k, q)
        
        return att
        
    def attention(self, v, k, q):
        
        d_k = q.shape[-1]
        
        att= torch.matmul(
            q, k.transpose(-2, -1)
        )/math.sqrt(d_k)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        return torch.matmul(att, v)

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FC(nn.Module):
    def __init__(self, in_size=256, out_size=256, dropout_r=0.15):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.linear = nn.Linear(in_size, out_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)














class SimpleRNN_SEED(nn.Module):
    def __init__(self, h_size, n_layer, in_size, f_dim, b_first = False, bidir = False):
        super(SimpleRNN_SEED, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.feat_dim = f_dim
        
        self.batch_first = b_first
        self.bidirectional = bidir

        self.b_n1 = nn.BatchNorm2d(self.feat_dim)

        self.RNN_pupil = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_dispersion = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_fixation = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_sacccade = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)


        self.fc_p = nn.Sequential(
            nn.Linear(12*h_size, self.hidden_size),
            nn.ReLU(True)
            )
        
        self.fc_d = nn.Sequential(
            nn.Linear(4*h_size, self.hidden_size),
            nn.ReLU(True)
            )

        self.fc_f = nn.Sequential(
            nn.Linear(2*h_size, self.hidden_size),
            nn.ReLU(True)
            )
        
        self.fc_s = nn.Sequential(
            nn.Linear(4*h_size, self.hidden_size),
            nn.ReLU(True)
            )

        self.fc = nn.Linear(4*self.hidden_size, self.hidden_size)

        self.b_n2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]
        
        x = self.b_n1(x.permute(0,2,1).view(x.shape[0], self.feat_dim, 1, -1 ))[:,:,0].permute(2, 0, 1)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()
        
        x_p, _  = self.RNN_pupil(x[:12], h0)
        x_d, _  = self.RNN_dispersion(x[12:16], h0)
        x_f, _  = self.RNN_fixation(x[16:18], h0)
        x_s, _  = self.RNN_sacccade(x[18:22], h0)

        x_p = self.fc_p(x_p.permute(1, 0, 2).reshape(self.batch_size, -1))
        x_d = self.fc_d(x_d.permute(1, 0, 2).reshape(self.batch_size, -1))
        x_f = self.fc_f(x_f.permute(1, 0, 2).reshape(self.batch_size, -1))
        x_s = self.fc_s(x_s.permute(1, 0, 2).reshape(self.batch_size, -1))

        x = torch.cat((x_p, x_d, x_f, x_s), dim=1)

        x = self.fc(x)
        
        x = self.b_n2(x)
        return x

class SimpleRNN_DEAP(nn.Module):
    def __init__(self, h_size, n_layer, in_size, f_dim, b_first = False, bidir = False, device=0):
        super(SimpleRNN_DEAP, self).__init__()

        self.device = device
        self.hidden_size = h_size
        self.num_layers = n_layer
        self.feat_dim = f_dim
        
        self.batch_first = b_first
        self.bidirectional = bidir

        self.b_n1 = nn.BatchNorm2d(self.feat_dim)

        self.RNN_EOG = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional ).cuda(device)
        self.RNN_EMG = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional ).cuda(device)
        self.RNN_Phy = nn.RNN(self.feat_dim, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional ).cuda(device)
        
        self.fc_o = nn.Sequential(
            nn.Linear(2*h_size, self.hidden_size),
            nn.ReLU(True)
            )
        
        self.fc_m = nn.Sequential(
            nn.Linear(2*h_size, self.hidden_size),
            nn.ReLU(True)
            )

        self.fc_p = nn.Sequential(
            nn.Linear(4*h_size, self.hidden_size),
            nn.ReLU(True)
            )

        self.fc = nn.Linear(3*self.hidden_size, self.hidden_size)

        self.b_n2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]
        
        x = self.b_n1(x.permute(0,2,1).view(x.shape[0], self.feat_dim, 1, -1 ))[:,:,0].permute(2, 0, 1)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(self.device)
        
        x_o, _  = self.RNN_EOG(x[:2], h0)
        x_m, _  = self.RNN_EMG(x[2:4], h0)
        x_p, _  = self.RNN_Phy(x[4:8], h0)

        x_o = self.fc_o(x_o.permute(1, 0, 2).reshape(self.batch_size, -1))
        x_m = self.fc_m(x_m.permute(1, 0, 2).reshape(self.batch_size, -1))
        x_p = self.fc_p(x_p.permute(1, 0, 2).reshape(self.batch_size, -1))

        x = torch.cat((x_o, x_m, x_p), dim=1)

        x = self.fc(x)
        x = self.b_n2(x)
        
        return x









class RegionRNN(nn.Module):

    def __init__(self, h_size, n_layer, in_size, f_dim, b_first = False, bidir = False):
        super(RegionRNN, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = 5
        self.feat_dim = f_dim

        self.dict = {'Fr1': np.array([0, 3, 8, 7, 6, 5]), 'Fr2': np.array([ 2,  4, 10, 11, 12, 13]), 'Tp1': np.array([14, 23, 32, 41, 50]), 
        'Tp2': np.array([22, 31, 40, 49, 56]), 'Cn1': np.array([15, 16, 17, 26, 25, 24, 33, 34, 35]), 'Cn2': np.array([21, 20, 19, 28, 29, 30, 39, 38, 37]), 
        'Pr1': np.array([42, 43, 44, 52, 51]), 'Pr2': np.array([48, 47, 46, 54, 55]), 'Oc1': np.array([58, 57]), 'Oc2': np.array([60, 61])}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_fL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_fR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_f = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_tL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_tR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_t = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_pL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_pR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_p = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_oL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_oR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_o = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        
        self.fc_f = nn.Sequential(
            nn.Linear(6*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_t = nn.Sequential(
            nn.Linear(5*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_p = nn.Sequential(
            nn.Linear(5*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_o = nn.Sequential(
            nn.Linear(2*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.b_n1 = nn.BatchNorm2d(self.feat_dim)
        self.b_n2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]
        
        x = self.b_n1(x.permute(0,2,1).view(x.shape[0], self.feat_dim, 1, -1 ))[:,:,0].permute(0,2,1)
        
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()
        
        k = list(self.dict.keys())

        fr_l = x[:, self.dict[k[0]]].permute(1, 0, 2)
        fr_r = x[:, self.dict[k[1]]].permute(1, 0, 2)

        tp_l = x[:, self.dict[k[2]]].permute(1, 0, 2)
        tp_r = x[:, self.dict[k[3]]].permute(1, 0, 2)

        p_l = x[:, self.dict[k[6]]].permute(1, 0, 2)
        p_r = x[:, self.dict[k[7]]].permute(1, 0, 2)

        o_l = x[:, self.dict[k[8]]].permute(1, 0, 2)
        o_r = x[:, self.dict[k[9]]].permute(1, 0, 2)

        x_fl, _ = self.RNN_fL(fr_l, h0)
        x_fr, _ = self.RNN_fR(fr_r, h0)

        x_tl, _ = self.RNN_tL(tp_l, h0)
        x_tr, _ = self.RNN_tR(tp_r, h0)

        x_pl, _ = self.RNN_tL(p_l, h0)
        x_pr, _ = self.RNN_tR(p_r, h0)

        x_ol, _ = self.RNN_oL(o_l, h0)
        x_or, _ = self.RNN_oR(o_r, h0)

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _  = self.RNN_f(x_f, h0)
        x_t, _  = self.RNN_f(x_t, h0)
        x_p, _  = self.RNN_p(x_p, h0)
        x_o, _  = self.RNN_o(x_o, h0)

        x_f = x_f.permute(1, 0, 2)
        x_t = x_t.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        x_o = x_o.permute(1, 0, 2)
        

        x = torch.cat((self.fc_f(x_f.reshape(self.batch_size, -1)), self.fc_t(x_t.reshape(self.batch_size, -1)), 
            self.fc_p(x_p.reshape(self.batch_size, -1)), self.fc_o(x_o.reshape(self.batch_size, -1))), dim=1)

        x = self.b_n2(x)
        #x = self.fc_f(x_f.reshape(self.batch_size, -1))  +  self.fc_t(x_t.reshape(self.batch_size, -1)) + self.fc_p(x_p.reshape(self.batch_size, -1)) + self.fc_o(x_o.reshape(self.batch_size, -1))
        #x = torch.cat((self.fc_f(x_f.reshape(self.batch_size, -1)), self.fc_t(x_t.reshape(self.batch_size, -1))), dim=1)
        x = x.reshape(self.batch_size, -1)
        #x = self.fc(x.reshape(self.batch_size, -1))
        return x
    
class RegionRNN_PhyDAA( nn.Module ):
    def __init__(self, h_size, n_layer, in_size, f_dim, b_first=False, bidir=False):
        super( RegionRNN_PhyDAA, self ).__init__()

        
        self.feat_dim = f_dim

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = in_size

        self.dict = {'Fr1': np.array( [0, 2, 3] ), 'Fr2': np.array( [30, 28, 29] ),
                     'Tp1': np.array( [4, 8, 9, 13] ),'Tp2': np.array( [25, 24, 19, 18] ),
                     'Cn1': np.array( [5, 6, 7, 10, 11] ), 'Cn2': np.array( [26, 27, 23, 20, 21] ),
                     'Pr1': np.array( [12] ), 'Pr2': np.array( [17] ),
                     'Oc1': np.array( [14] ), 'Oc2': np.array( [16] )}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_fL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_fR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_f = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_tL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_tR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_t = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_pL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_pR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_p = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_oL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_oR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_o = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.fc_f = nn.Sequential(
            nn.Linear( 3 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_t = nn.Sequential(
            nn.Linear( 4 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_p = nn.Sequential(
            nn.Linear( 1 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_o = nn.Sequential(
            nn.Linear( 1 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.b_n1 = nn.BatchNorm2d( self.feat_dim )
        self.b_n2 = nn.BatchNorm1d( 64 )

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.b_n1( x.permute( 0, 2, 1 ).reshape( x.shape[0], self.feat_dim, 1, -1 ) )[:, :, 0].permute( 0, 2, 1 )

        h0 = torch.zeros( self.num_layers, self.batch_size, self.hidden_size ).cuda()
        k = list( self.dict.keys() )

        fr_l = x[:, self.dict[k[0]]].permute( 1, 0, 2 )
        fr_r = x[:, self.dict[k[1]]].permute( 1, 0, 2 )

        tp_l = x[:, self.dict[k[2]]].permute( 1, 0, 2 )
        tp_r = x[:, self.dict[k[2]]].permute( 1, 0, 2 )

        p_l = x[:, self.dict[k[6]]].permute( 1, 0, 2 )
        p_r = x[:, self.dict[k[7]]].permute( 1, 0, 2 )

        o_l = x[:, self.dict[k[8]]].permute( 1, 0, 2 )
        o_r = x[:, self.dict[k[9]]].permute( 1, 0, 2 )

        x_fl, _ = self.RNN_fL( fr_l, h0 )
        x_fr, _ = self.RNN_fR( fr_r, h0 )

        x_tl, _ = self.RNN_tL( tp_l, h0 )
        x_tr, _ = self.RNN_tR( tp_r, h0 )

        x_pl, _ = self.RNN_tL( p_l, h0 )
        x_pr, _ = self.RNN_tR( p_r, h0 )

        x_ol, _ = self.RNN_oL( o_l, h0 )
        x_or, _ = self.RNN_oR( o_r, h0 )

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _ = self.RNN_f( x_f, h0 )
        x_t, _ = self.RNN_f( x_t, h0 )
        x_p, _ = self.RNN_p( x_p, h0 )
        x_o, _ = self.RNN_o( x_o, h0 )

        x_f = x_f.permute( 1, 0, 2 )
        x_t = x_t.permute( 1, 0, 2 )
        x_p = x_p.permute( 1, 0, 2 )
        x_o = x_o.permute( 1, 0, 2 )

        x = torch.cat(
            (self.fc_f( x_f.reshape( self.batch_size, -1 ) ), self.fc_t( x_t.reshape( self.batch_size, -1 ) ),
             self.fc_p( x_p.reshape( self.batch_size, -1 ) ), self.fc_o( x_o.reshape( self.batch_size, -1 ) )), dim=1 )

        x = self.b_n2( x )
        x = x.reshape( self.batch_size, -1 )

        return x


class RegionRNN_DEAP( nn.Module ):
    def __init__(self, h_size, n_layer, in_size, f_dim, b_first=False, bidir=False, device=0):
        super( RegionRNN_DEAP, self ).__init__()

        self.device = device
        self.feat_dim = f_dim

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = in_size

        self.dict = {'Fr1': np.array( [0, 1, 2, 3] ), 'Fr2': np.array( [16, 17, 19, 20] ),
                     'Tp1': np.array( [7, 11] ),'Tp2': np.array( [25, 29] ),
                     'Cn1': np.array( [4, 5, 6, 8, 9] ), 'Cn2': np.array( [21, 22, 24, 26, 27] ),
                     'Pr1': np.array( [10, 12] ), 'Pr2': np.array( [28, 30] ),
                     'Oc1': np.array( [12, 13] ), 'Oc2': np.array( [30, 31] )}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_fL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)
        self.RNN_fR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)

        self.RNN_f = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional ).cuda(device)

        self.RNN_tL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)
        self.RNN_tR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)

        self.RNN_t = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional ).cuda(device)

        self.RNN_cL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)
        self.RNN_cR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)

        self.RNN_c = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional ).cuda(device)

        self.RNN_pL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)
        self.RNN_pR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)

        self.RNN_p = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional ).cuda(device)

        self.RNN_oL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)
        self.RNN_oR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional ).cuda(device)

        self.RNN_o = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional ).cuda(device)

        self.fc_f = nn.Sequential(
            nn.Linear( 4 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_t = nn.Sequential(
            nn.Linear( 2 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_c = nn.Sequential(
            nn.Linear( 5 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_p = nn.Sequential(
            nn.Linear( 2 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_o = nn.Sequential(
            nn.Linear( 2 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc = nn.Linear(5*16, 64)

        self.b_n1 = nn.BatchNorm2d( self.feat_dim )
        self.b_n2 = nn.BatchNorm1d( 64 )

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.b_n1( x.permute( 0, 2, 1 ).reshape( x.shape[0], self.feat_dim, 1, -1 ) )[:, :, 0].permute( 0, 2, 1 )

        h0 = torch.zeros( self.num_layers, self.batch_size, self.hidden_size ).cuda(self.device)
        k = list( self.dict.keys() )

        fr_l = x[:, self.dict[k[0]]].permute( 1, 0, 2 )
        fr_r = x[:, self.dict[k[1]]].permute( 1, 0, 2 )

        tp_l = x[:, self.dict[k[2]]].permute( 1, 0, 2 )
        tp_r = x[:, self.dict[k[3]]].permute( 1, 0, 2 )

        c_l = x[:, self.dict[k[4]]].permute( 1, 0, 2 )
        c_r = x[:, self.dict[k[5]]].permute( 1, 0, 2 )

        p_l = x[:, self.dict[k[6]]].permute( 1, 0, 2 )
        p_r = x[:, self.dict[k[7]]].permute( 1, 0, 2 )

        o_l = x[:, self.dict[k[8]]].permute( 1, 0, 2 )
        o_r = x[:, self.dict[k[9]]].permute( 1, 0, 2 )

        x_fl, _ = self.RNN_fL( fr_l, h0 )
        x_fr, _ = self.RNN_fR( fr_r, h0 )

        x_tl, _ = self.RNN_tL( tp_l, h0 )
        x_tr, _ = self.RNN_tR( tp_r, h0 )

        x_cl, _ = self.RNN_cL( c_l, h0 )
        x_cr, _ = self.RNN_cL( c_r, h0 )

        x_pl, _ = self.RNN_tL( p_l, h0 )
        x_pr, _ = self.RNN_tR( p_r, h0 )

        x_ol, _ = self.RNN_oL( o_l, h0 )
        x_or, _ = self.RNN_oR( o_r, h0 )

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_c = x_cr - x_cl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _ = self.RNN_f( x_f, h0 )
        x_t, _ = self.RNN_f( x_t, h0 )
        x_c, _ = self.RNN_c( x_c, h0 )
        x_p, _ = self.RNN_p( x_p, h0 )
        x_o, _ = self.RNN_o( x_o, h0 )

        x_f = x_f.permute( 1, 0, 2 )
        x_t = x_t.permute( 1, 0, 2 )
        x_c = x_c.permute( 1, 0, 2 )
        x_p = x_p.permute( 1, 0, 2 )
        x_o = x_o.permute( 1, 0, 2 )

        x = torch.cat(
            (self.fc_f( x_f.reshape( self.batch_size, -1 ) ), self.fc_t( x_t.reshape( self.batch_size, -1 ) ),
            self.fc_c(x_c.reshape(self.batch_size, -1) ), self.fc_p( x_p.reshape( self.batch_size, -1 ) ), 
            self.fc_o( x_o.reshape( self.batch_size, -1 ) )), dim=1 )

        x = self.fc( x )
        x = self.b_n2( x )
        x = x.reshape( self.batch_size, -1 )

        return x

class RegionRNN_VIG( nn.Module ):
    def __init__(self, h_size, n_layer, in_size, f_dim, b_first=False, bidir=False):
        super( RegionRNN_VIG, self ).__init__()

        
        self.feat_dim = f_dim

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = in_size

        self.dict = {'Tp1': np.array( [0, 2, 4] ),'Tp2': np.array( [1, 3, 5] ),
                     'Pr1': np.array( [6, 8, 11] ), 'Pr2': np.array( [7, 10, 13] ),
                     'Oc1': np.array( [14] ), 'Oc2': np.array( [16] )}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_tL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_tR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_t = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_pL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_pR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_p = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )

        self.RNN_oL = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )
        self.RNN_oR = nn.RNN( self.input_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                              bidirectional=self.bidirectional )

        self.RNN_o = nn.RNN( self.hidden_size, self.hidden_size, self.num_layers, batch_first=self.batch_first,
                             bidirectional=self.bidirectional )


        self.fc_t = nn.Sequential(
            nn.Linear( 3 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_p = nn.Sequential(
            nn.Linear( 3 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_o = nn.Sequential(
            nn.Linear( 1 * self.hidden_size, 16 ),
            nn.ReLU(),
        )

        self.fc_final = nn.Sequential(
            nn.Linear( 3*16, 4*16),
            nn.ReLU(),
        )

        self.b_n1 = nn.BatchNorm2d( self.feat_dim )
        self.b_n2 = nn.BatchNorm1d( 64 )

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.b_n1( x.permute( 0, 2, 1 ).reshape( x.shape[0], self.feat_dim, 1, -1 ) )[:, :, 0].permute( 0, 2, 1 )

        h0 = torch.zeros( self.num_layers, self.batch_size, self.hidden_size ).cuda()
        k = list( self.dict.keys() )

        tp_l = x[:, self.dict[k[0]]].permute( 1, 0, 2 )
        tp_r = x[:, self.dict[k[1]]].permute( 1, 0, 2 )

        p_l = x[:, self.dict[k[2]]].permute( 1, 0, 2 )
        p_r = x[:, self.dict[k[3]]].permute( 1, 0, 2 )

        o_l = x[:, self.dict[k[4]]].permute( 1, 0, 2 )
        o_r = x[:, self.dict[k[5]]].permute( 1, 0, 2 )

        x_tl, _ = self.RNN_tL( tp_l, h0 )
        x_tr, _ = self.RNN_tR( tp_r, h0 )

        x_pl, _ = self.RNN_tL( p_l, h0 )
        x_pr, _ = self.RNN_tR( p_r, h0 )

        x_ol, _ = self.RNN_oL( o_l, h0 )
        x_or, _ = self.RNN_oR( o_r, h0 )

        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_t, _ = self.RNN_t( x_t, h0 )
        x_p, _ = self.RNN_p( x_p, h0 )
        x_o, _ = self.RNN_o( x_o, h0 )

        x_t = x_t.permute( 1, 0, 2 )
        x_p = x_p.permute( 1, 0, 2 )
        x_o = x_o.permute( 1, 0, 2 )

        x = torch.cat(
            (self.fc_t( x_t.reshape( self.batch_size, -1 ) ),
             self.fc_p( x_p.reshape( self.batch_size, -1 ) ), self.fc_o( x_o.reshape( self.batch_size, -1 ) )), dim=1 )

        x = self.fc_final( x )
        x = self.b_n2( x )
        x = x.reshape( self.batch_size, -1 )

        return x



class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)