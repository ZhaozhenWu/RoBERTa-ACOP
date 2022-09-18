# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import copy
import numpy as np
from transformers.modeling_bert import BertLayer
BertLayerNorm = torch.nn.LayerNorm


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


class selfalignment(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, bias=False):
        super(selfalignment, self).__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(0.1)
        self.linear = torch.nn.Linear(in_features, in_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(in_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, text1, textmask1, textmask2):  # b,s1,h;b,s2,h;b,s1;b,s2
        logit = torch.matmul(self.linear(text), text1.transpose(1, 2))  # b,s1,s2
        textmask11 = textmask1.unsqueeze(-1)  # b,s1,1
        textmask22 = textmask2.unsqueeze(1)  # b,1,s2
        masked = textmask11 * textmask22  # b,s1,s2
        masked = (1 - masked) * (-10000.0)
        logits = torch.softmax(logit + masked, -1)  # b,s1,s2
        logits1 = torch.softmax(logit + masked, -2)  # b,s1,s2
        output = torch.matmul(logits, text1)
        output = output * textmask1.unsqueeze(-1)  # b,s1,h
        output1 = torch.matmul(logits1.transpose(1, 2), text)
        output1 = output1 * textmask2.unsqueeze(-1)  # b,s2,h
        return output + text, output1 + text1, logits * textmask1.unsqueeze(-1)


def init_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.01)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class simpleGraphConvolutionalignment(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, edge_size, bias=False):
        super(simpleGraphConvolutionalignment, self).__init__()
        self.K = 3
        self.bertlayers = torch.nn.ModuleList([BertLayer(config) for _ in range(self.K)])
        self.norm = torch.nn.LayerNorm(out_features)
        self.edge_vocab = torch.nn.Embedding(edge_size, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(0.1)
        self.mutual = selfalignment(in_features)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.linear = torch.nn.Linear(out_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def renorm(self, adj1, adj2):
        adj = adj1 * adj2
        adj = adj / (adj.sum(-1).unsqueeze(-1) + 1e-10)
        return adj

    def forward(self, text, adj1, adj2, edge1, edge2, textmask):
        adj = adj1 + adj2
        adj[adj >= 1] = 1

        extended_attention_mask = textmask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attss = [adj]
        if (self.in_features != self.out_features):
            out = torch.relu(torch.matmul(text, self.weight))
            output = out
            outs = out
        else:
            out = text
            output = out
            outs = out
        for i in range(self.K):
            outs = self.bertlayers[i](outs, attention_mask=extended_attention_mask)[0]  # get transformer output
            teout = self.linear(output)
            denom1 = torch.sum(adj, dim=2, keepdim=True) + 0.0000001
            output = self.dropout(torch.relu(torch.matmul(adj, teout) / denom1)) + output  # get GCN output

            outs, output, att = self.mutual(outs, output, textmask, textmask)  # biaffine operation
        return outs, attss  # b,s,h


def length2mask(length, maxlength):
    size = list(length.size())
    length = length.unsqueeze(-1).repeat(*([1] * len(size) + [maxlength])).long()
    ran = torch.arange(maxlength).cuda()
    ran = ran.expand_as(length)
    mask = ran < length
    return mask.float().cuda()


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


class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.gc2 = GraphConvolution(2 * opt.hidden_dim, 2 * opt.hidden_dim)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output


class mutualatt(nn.Module):
    def __init__(self, hidden_size):
        super(mutualatt, self).__init__()
        self.linear2 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, in1, text, textmask):
        length = text.size(1)
        in1 = in1.unsqueeze(1).repeat(1, length, 1)
        att = self.linear1(torch.relu(self.linear2(torch.cat([in1, text], -1)))).squeeze(-1)
        att = torch.softmax(att + textmask, -1).unsqueeze(-2)

        context = torch.matmul(att, text).squeeze(1)
        return context, att


class ASBIGCN(nn.Module):
    def __init__(self, opt):
        super(ASBIGCN, self).__init__()
        self.opt = opt
        self.mul1 = mutualatt(2 * opt.hidden_dim)

        self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.gc = simpleGraphConvolutionalignment(2 * opt.hidden_dim, 2 * opt.hidden_dim, opt.edge_size, bias=True)
        self.fc = nn.Linear(10 * opt.hidden_dim, opt.polarities_dim)

        self.text_embed_dropout = nn.Dropout(0.1)
        self.linear1 = torch.nn.Linear(2 * opt.hidden_dim, 2 * opt.hidden_dim, bias=False)

    def forward(self, inputs, mixed=True):
        text_indices, span_indices, tran_indices, adj1, adj2, edge1, edge2 = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        outputs = self.bert(text_indices, attention_mask=length2mask(text_len, text_indices.size(1)))[0]
        output = outputs[:, 1:, :]
        max_len = max([len(item) for item in tran_indices])
        text_len = torch.Tensor([len(item) for item in tran_indices]).long().cuda()
        tmps = torch.zeros(text_indices.size(0), max_len, 768).float().cuda()
        for i, spans in enumerate(tran_indices):
            for j, span in enumerate(spans):
                tmps[i, j] = torch.sum(output[i, span[0]:span[1]], 0)

        text = self.text_embed_dropout(tmps)

        text, (hout, _) = self.text_lstm(text, text_len.cpu())
        x = text

        hout = torch.transpose(hout, 0, 1)
        hout = hout.reshape(hout.size(0), -1)

        text, self.attss = self.gc(text, adj1, adj2, edge1, edge2, length2mask(text_len, max_len))
        spanlen = max([len(item) for item in span_indices])
        tmp = torch.zeros(text_indices.size(0), spanlen, 2 * self.opt.hidden_dim).float().cuda()
        tmp1 = torch.zeros(text_indices.size(0), spanlen, 2 * self.opt.hidden_dim).float().cuda()
        for i, spans in enumerate(span_indices):
            for j, span in enumerate(spans):
                tmp[i, j] = torch.sum(text[i, span[0]:span[1]], -2)

                tmp1[i, j] = torch.sum(x[i, span[0]:span[1]], -2)
        v1, _ = torch.max(text, -2)
        v2, _ = torch.max(x, -2)

        output = self.fc(torch.cat(
            [hout, tmp[:, 0, :], tmp1[:, 0, :], tmp[:, 0, :] * tmp1[:, 0, :], torch.abs(tmp[:, 0, :] - tmp1[:, 0, :])],
            -1))

        return output
