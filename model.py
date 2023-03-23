import torch
import torch.nn as nn



class EmbeddingLayer(nn.Module):
    def __init__(self):
        pass
        # implement here patchify method


class Transformer(nn.Module):
    def __init__(self):
        pass


class SelfAttention(nn.Module):
    def __init__(self):
        pass


class MLPHead(nn.Module):
    def __init__(self):
        pass
        # final module here


class SelfAttention(nn.Module):
    def __init__(self, dimx, dimq, n_heads):
        super(SelfAttention, self).__init__()
        self.dimx = dimx
        self.dimq = dimq
        self.n_heads = n_heads
        self.softmax = nn.Softmax(dim=3)

        # initialize weight matrices for query, key, and value
        self.W_q = nn.Linear(dimx, dimq*n_heads, bias=False).weight
        self.W_k = nn.Linear(dimx, dimq*n_heads, bias=False).weight
        self.W_v = nn.Linear(dimx, dimx*n_heads, bias=False).weight

        # initialize re-mapping weight matrix (from n_heads*dimx to dimx)
        self.W_r = nn.Linear(dimx*n_heads, dimx, bias=False).weight

    def forward(self, X):
        X = X.transpose(1, 2)

        # get query, key and value matrices
        Q = self.W_q @ X
        K = self.W_k @ X
        V = self.W_v @ X

        # split into heads
        n_samples = X.shape[0]
        Q = Q.reshape(n_samples, self.n_heads, self.dimq, Q.shape[-1])
        K = K.reshape(n_samples, self.n_heads, self.dimq, K.shape[-1])
        V = V.reshape(n_samples, self.n_heads, self.dimx, V.shape[-1])

        # get attention matrix Z
        Z = self.softmax(Q.transpose(2, 3) @ K / self.dimq**0.5)

        # get result matrix R i.e. linear combination of columns in V
        R = V @ Z.transpose(2, 3)

        # concat heads
        R = R.reshape(R.shape[0], -1, R.shape[-1])

        # go back to original dimension i.e. dimx
        R = self.W_r @ R
        return R.transpose(1, 2)
