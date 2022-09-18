# -*- coding: utf-8 -*-

import argparse
import math
import os
import random

import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from transformers import RobertaModel, BertModel

from bucket_iterator import BucketIterator
from data_utils import *
from models import LSTM, ASCNN, ASGCN, ASBIGCN, RoBERTaACOP, RoBERTaPURE
from optimization import AdamW, WarmupLinearSchedule

roberta = RobertaModel.from_pretrained('D:\pretrained-models\\roberta')
bert = BertModel.from_pretrained('D:\pretrained-models\\bert_base_uncased')


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, window=opt.window, is_window=opt.is_window)
        # dump data
        fout = open('load_data/' + opt.dataset + '.order.pkl', 'wb')
        pickle.dump(absa_dataset, fout)
        fout.close()

        # load data
        # with open('load_data/' + opt.dataset + '.order.pkl', 'rb') as f:
        #     absa_dataset = pickle.load(f)

        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)

        self.model = opt.model_class(opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, scheduler):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0

        if self.opt.is_window:
            result_file = 'log/' + self.opt.model_name + '_' + self.opt.dataset \
                          + '_window' + str(self.opt.window) + '_result.txt'
        else:
            result_file = 'log/' + self.opt.model_name + '_' + self.opt.dataset + '_result.txt'
        with open(result_file, 'w', encoding='utf-8') as f_out:
            for epoch in range(self.opt.num_epoch):
                print('>' * 100)
                print('epoch: ', epoch)
                n_correct, n_total = 0, 0
                increase_flag = False
                for i_batch, sample_batched in enumerate(self.train_data_loader):
                    global_step += 1

                    # switch model to training mode, clear gradient accumulators
                    self.model.train()

                    optimizer.zero_grad()
                    inputs = [
                        sample_batched[col].to(self.opt.device) if type(sample_batched[col]) != list else
                        sample_batched[
                            col] for col in self.opt.inputs_cols]
                    targets = sample_batched['polarity'].to(self.opt.device)
                    targets_order = sample_batched['order'].to(self.opt.device)

                    outputs, outputs_order = self.model(inputs)
                    # handle loss
                    loss = (criterion(outputs, targets) + criterion(outputs_order, targets_order)) / 2 \
                        if self.opt.model_name == 'roberta_acop' else criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # test
                    if global_step % self.opt.log_step == 0:
                        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                        n_total += len(outputs)
                        train_acc = n_correct / n_total

                        test_acc, test_f1 = self._evaluate_acc_f1()
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                            if self.opt.save and not self.opt.is_window:
                                torch.save(self.model.state_dict(),
                                           'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.acc.pkl')
                        if test_f1 > max_test_f1:
                            increase_flag = True
                            max_test_f1 = test_f1
                            if self.opt.save and test_f1 > self.global_f1:
                                self.global_f1 = test_f1
                                if not self.opt.is_window:
                                    torch.save(self.model.state_dict(),
                                               'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.f1.pkl')
                                print('>>> best model saved.')
                        f_out.write('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}\n'.format(loss.item(),
                                                                                                            train_acc,
                                                                                                            test_acc,
                                                                                                            test_f1))
                        print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(),
                                                                                                    train_acc,
                                                                                                    test_acc, test_f1))
                if not increase_flag:
                    continue_not_increase += 1
                    if continue_not_increase >= 4:
                        print('early stop.')
                        break
                else:
                    continue_not_increase = 0
            return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [
                    t_sample_batched[col].to(opt.device) if type(t_sample_batched[col]) != list else t_sample_batched[
                        col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs, _ = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return test_acc, f1

    def _evaluate_acc_f1_some_work(self):
        # switch model to evaluation mode
        self.model.eval()
        import os
        if not os.path.exists('results/' + self.opt.model_name + 're' + self.opt.dataset):
            os.makedirs('results/' + self.opt.model_name + 're' + self.opt.dataset)
            os.makedirs('results/' + self.opt.model_name + 're' + self.opt.dataset + '/right')
            os.makedirs('results/' + self.opt.model_name + 're' + self.opt.dataset + '/false')
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        all_results = []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [
                    t_sample_batched[col].to(opt.device) if type(t_sample_batched[col]) != list else
                    t_sample_batched[
                        col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs, _ = self.model(t_inputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                for i in range(t_sample_batched['polarity'].size(0)):
                    tmp_dict = {'text': t_sample_batched['text'][i],
                                'true_label': t_targets[i].cpu().data.tolist(),
                                'predict_label': torch.argmax(t_outputs, -1)[i].cpu().data.tolist()}
                    tmp_dict['is_right'] = tmp_dict['true_label'] == tmp_dict['predict_label']
                    all_results.append(tmp_dict)
        with open('results/' + self.opt.model_name + self.opt.dataset + '.acc.false', 'w', encoding='utf-8') as f:
            for index, item in enumerate(all_results):
                if not item['is_right']:
                    f.write(item['text'] + '\n')
                    f.write(str(item['true_label']) + '\n')
                    f.write(str(item['predict_label']) + '\n')
                    f.write(str(item['is_right']) + '\n')

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        if not os.path.exists('log/'):
            os.mkdir('log/')

        if self.opt.is_window:
            val_file = 'log/' + self.opt.model_name + '_' + self.opt.dataset \
                       + '_window' + str(self.opt.window) + '_val.txt'
        else:
            val_file = 'log/' + self.opt.model_name + '_' + self.opt.dataset + '_val.txt'
        f_out = open(val_file, 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1))
            self._reset_params()
            self.model.roberta = roberta
            self.model.cuda()

            if self.opt.mode == "train":
                # parameters = [p for name, p in self.model.named_parameters()]
                named_params = [(name, p) for name, p in self.model.named_parameters()]

                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                     'weight_decay': 0.01},
                    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

                # optimizer = BertAdam(optimizer_grouped_parameters, lr=0.00005, warmup=0.1,
                #                      t_total=self.train_data_loader.batch_len * 20)
                optimizer = AdamW(optimizer_grouped_parameters, lr=2e-6, eps=1e-8)
                scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.train_data_loader.batch_len * 20 * 0.06,
                                                 t_total=self.train_data_loader.batch_len * 20)
                # optimizer_grouped_parameters = [
                #     {'params': parameters, 'weight_decay': 0.01}
                # ]

                # optimizer1 = BertAdam(optimizer_grouped_parameters, lr=0.00005, warmup=0.1,
                #                       t_total=self.train_data_loader.batch_len * 20)
                max_test_acc, max_test_f1 = self._train(criterion, optimizer, scheduler)
                print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
                f_out.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_acc, max_test_f1))
                max_test_acc_avg += max_test_acc
                max_test_f1_avg += max_test_f1
                print('#' * 100)
                print("max_test_acc_avg:", max_test_acc_avg / repeats)
                print("max_test_f1_avg:", max_test_f1_avg / repeats)
                #        torch.save(self.model.state_dict(),self.opt.model_name+'_'+self.opt.dataset+'.pth')
            else:
                self.model.load_state_dict(
                    torch.load('state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl'))
                test_acc, test_f1 = self._evaluate_acc_f1_some_work()
                print("max_test_acc_avg:", test_acc / repeats)
                print("max_test_f1_avg:", test_f1 / repeats)
        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='roberta_acop', type=str, help='roberta_acop, roberta_pure')
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-6, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    # parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--is_window', default=True, type=str)
    parser.add_argument('--window', default=6, type=str)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
        'astcn': ASGCN,
        'asbi': ASBIGCN,
        'roberta_pure': RoBERTaPURE,
        'roberta_acop': RoBERTaACOP,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'roberta_pure': ['text', 'text_indices', 'text_indices_trans', 'trans', 'aspect_indices'],
        'roberta_acop': ['text', 'text_indices', 'text_indices_trans', 'trans', 'aspect_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'asbi': ['text_indices', 'span_indices', 'tran_indices', 'dependency_graph', 'dependency_graph1',
                 'dependency_graph2', 'dependency_graph3'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
