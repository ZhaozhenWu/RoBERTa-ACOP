# -*- coding: utf-8 -*-
import pickle
import re

import spacy
from spacy.attrs import ORTH
import tqdm
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('D:\pretrained-models\\roberta')
# tokenizer = BertTokenizer.from_pretrained('D:\pretrained-models\\bert_base_uncased')

nlp = spacy.load('en_core_web_sm')


def tokenize_spacy(text):
    text = text.strip()
    text = re.sub(r' {2,}', ' ', text)
    nlp.tokenizer.add_special_case("</s>", [{ORTH: "</s>"}])
    nlp.tokenizer.add_special_case("-rrb-", [{ORTH: "-rrb-"}])
    nlp.tokenizer.add_special_case("-lrb-", [{ORTH: "-lrb-"}])
    document = nlp(text)
    return [token.text for token in document]


def text_to_sequence(text):
    text = text.lower().strip()
    words = tokenize_spacy(text)
    trans = []
    real_word_indices = []

    for word in words:
        word_indices = tokenizer(' ' + word)['input_ids'] if word != '</s>' else tokenizer(word)['input_ids']
        temp_len = len(real_word_indices)
        real_word_indices.extend(word_indices[1:-1])
        trans.append([temp_len, len(real_word_indices)])

    sequence = [0] + real_word_indices + [2]

    if len(sequence) == 0:
        print("error!")

    return sequence, trans


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_data__(fname, tokenizer):
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            lines = f.readlines()

        all_data = []
        for i in tqdm.tqdm(range(0, len(lines), 4)):
            # text_left = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            order = lines[i + 3].strip()
            text = lines[i].lower().strip()

            text_indices_trans, trans = text_to_sequence(text)

            aspect_indices = tokenizer(aspect)['input_ids']
            text_indices = tokenizer(' ' + text)['input_ids']

            polarity = int(polarity) + 1
            order = int(order)
            data = {
                'text': text.lower().strip(),
                'text_indices': text_indices,
                'text_indices_trans': text_indices_trans,
                'trans': trans,
                'aspect': aspect,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'order': order,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='rest14', window=1, is_window=False):
        print("preparing {0} dataset ...".format(dataset))
        if is_window:
            fname = {
                'twitter': {
                    'train': './datasets/twitter/train.raw.local.window' + str(window),
                    'test': './datasets/twitter/test.raw.order'
                },
                'rest14': {
                    'train': './datasets/semeval14/restaurant_train.raw.local.window' + str(window),
                    'test': './datasets/semeval14/restaurant_test.raw.order'
                },
                'lap14': {
                    'train': './datasets/semeval14/laptop_train.raw.local.window' + str(window),
                    'test': './datasets/semeval14/laptop_test.raw.order'
                },
                'rest15': {
                    'train': './datasets/semeval15/restaurant_train.raw.local.window' + str(window),
                    'test': './datasets/semeval15/restaurant_test.raw.order'
                },
                'rest16': {
                    'train': './datasets/semeval16/restaurant_train.raw.local.window' + str(window),
                    'test': './datasets/semeval16/restaurant_test.raw.order'
                }
            }
        else:
            fname = {
                'twitter': {
                    'train': './datasets/twitter/train.raw.order',
                    'test': './datasets/twitter/test.raw.order'
                },
                'rest14': {
                    'train': './datasets/semeval14/restaurant_train.raw.order',
                    'test': './datasets/semeval14/restaurant_test.raw.order'
                },
                'lap14': {
                    'train': './datasets/semeval14/laptop_train.raw.order',
                    'test': './datasets/semeval14/laptop_test.raw.order'
                },
                'rest15': {
                    'train': './datasets/semeval15/restaurant_train.raw.order',
                    'test': './datasets/semeval15/restaurant_test.raw.order'
                },
                'rest16': {
                    'train': './datasets/semeval16/restaurant_train.raw.order',
                    'test': './datasets/semeval16/restaurant_test.raw.order'
                }
            }

        self.tokenizer = tokenizer
        self.train_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['train'], self.tokenizer))
        # print("---------------------------------load datasets:" + fname[dataset]['train'])
        self.test_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['test'], self.tokenizer))


if __name__ == '__main__':
    print('debugging')
