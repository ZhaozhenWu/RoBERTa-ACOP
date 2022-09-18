# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Config:
    num_attention_heads = 1
    layer_norm_eps = 1e-20
    hidden_size = 200
    hidden_dropout_prob = 0.3
    intermediate_size = 200
    output_attentions = True
    attention_probs_dropout_prob = 0.3
    hidden_act = 'gelu'
    chunk_size_feed_forward = 0


config = Config()


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RoBERTaPURE(nn.Module):
    def __init__(self, opt):
        super(RoBERTaPURE, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(768, opt.polarities_dim)

    def forward(self, inputs):
        text_indices, text_indices_trans, trans, dep_graph, aspect_indices = inputs

        aspect_text = torch.cat([text_indices_trans, aspect_indices], -1)
        output = self.roberta(aspect_text)[1]

        outputs = self.fc1(output)
        return outputs, None
