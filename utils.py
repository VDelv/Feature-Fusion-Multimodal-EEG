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

import mne
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def rmse(x, y):
    return sum((x - y)**2)/x.shape[0]

def corr(x, y):
    corr = ((x-x.mean())*(y-y.mean())).sum()
    corr = corr/np.sqrt(((x-x.mean())**2).sum()*((y-y.mean())**2).sum())
    return corr

def freq_filter(sig, sampling_frequency=128, delta_lim=0.5, theta_lim=4, alpha_lim=8, beta_lim=14, gamma_lim=31):
    d = mne.filter.filter_data(sig, sfreq=sampling_frequency, l_freq=delta_lim, h_freq=theta_lim, fir_window='hann', filter_length='8s', verbose='CRITICAL')
    t = mne.filter.filter_data(sig, sfreq=sampling_frequency, l_freq=theta_lim, h_freq=alpha_lim, fir_window='hann', filter_length='8s', verbose='CRITICAL')
    a = mne.filter.filter_data(sig, sfreq=sampling_frequency, l_freq=alpha_lim, h_freq=beta_lim, fir_window='hann', filter_length='8s', verbose='CRITICAL')
    b = mne.filter.filter_data(sig, sfreq=sampling_frequency, l_freq=beta_lim, h_freq=gamma_lim, fir_window='hann', filter_length='8s', verbose='CRITICAL')
    g = mne.filter.filter_data(sig, sfreq=sampling_frequency, l_freq=gamma_lim, h_freq=50, fir_window='hann', filter_length='8s', verbose='CRITICAL')
    return np.stack((d, t, a, b, g), axis=0)

def entropy(sig):
    return np.log(2*np.pi*np.exp(1)*np.std(sig))/2

def gen_feat(sig, sampling_frequency=128, win_len=4):
    n_samples = sampling_frequency*win_len
    freq_bands = freq_filter(sig)
    feat_mat = np.zeros((freq_bands.shape[0], int(np.ceil(freq_bands.shape[1]/n_samples))))
    for b in range(freq_bands.shape[0]):
        for s in range(int(np.ceil(freq_bands.shape[1]/n_samples))):
            feat_mat[b,s] = entropy(freq_bands[b, s*n_samples:(s+1)*n_samples])
    return feat_mat

def feat_matrices(sig, n_bands=5, sampling_frequency=128):
    n_trial = sig.shape[0]
    n_chan = sig.shape[1]
    n_samples = int(np.ceil(sig.shape[2]/(4*sampling_frequency)))
    feat = np.zeros((n_trial, n_chan, n_bands, n_samples))

    for c in range(n_chan):
        for t in range(n_trial):
            feat[t, c] = gen_feat(sig[t, c])
    return feat
