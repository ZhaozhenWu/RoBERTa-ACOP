# -*- coding: utf-8 -*-
import itertools
import random
import re
import string

import spacy
import tqdm
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('D:\pretrained-models\\roberta')
nlp = spacy.load('en_core_web_sm')
punctuation_string = string.punctuation


def tokenize(text):
    text = text.strip()
    text = re.sub(r' {2,}', ' ', text)
    document = nlp(text)
    return [token.text for token in document]


def process_train(filename):
    nums = [0, 1, 2]

    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    output = filename.replace('.raw', '.raw.order')

    with open(output, 'w', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(0, len(lines), 3)):
            texts = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].lower().strip()
            texts.insert(1, aspect)
            for index, num in enumerate(itertools.permutations(nums)):
                temp = [texts[num[item]] for item in nums]
                target = " ".join(temp)
                f.write(target + "\n")
                f.write(aspect + "\n")
                f.write(polarity + "\n")
                if i == len(lines) - 3 and index == 5:
                    f.write(str(index))
                else:
                    f.write(str(index) + "\n")


def process_local_train(filename, window):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    output = filename.replace('.raw', '.raw.local.window' + str(window))

    with open(output, 'w', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(0, len(lines), 3)):
            texts = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            aspect_len = len(aspect.split(" "))
            polarity = lines[i + 2].lower().strip()
            texts.insert(1, aspect)
            # insert original text
            f.write(" ".join(texts) + "\n")
            f.write(aspect + "\n")
            f.write(polarity + "\n")
            f.write(str(0) + "\n")

            text_left_len = len(texts[0].split(" "))
            text_right_len = len(texts[2].split(" "))
            text_len = aspect_len + text_left_len + text_right_len
            # aspect index
            aspect_index_left = text_left_len
            aspect_index_right = aspect_index_left + aspect_len
            # handle local ACOP
            text_window_index_left = aspect_index_left - window if aspect_index_left - window > 0 else 0
            text_window_index_right = aspect_index_right + window \
                if aspect_index_right + window + 1 < text_len else text_len
            text_choice_list = list(range(text_window_index_left, text_window_index_right - aspect_len + 1, 1))
            text_choice_list.remove(aspect_index_left)
            context = texts[0].split(" ") + texts[2].split(" ")
            if len(text_choice_list) <= 5:
                index_choice = random.sample(text_choice_list, len(text_choice_list))
            else:
                index_choice = random.sample(text_choice_list, 5)
            for j, index in enumerate(index_choice):
                context.insert(index, aspect)
                target = " ".join(context)
                f.write(target + "\n")
                f.write(aspect + "\n")
                f.write(polarity + "\n")
                if i == len(lines) - 3 and j == 4:
                    f.write(str(j + 1))
                else:
                    f.write(str(j + 1) + "\n")
                context.pop(index)


def concat(texts, aspect):
    source = ''
    splitnum = 0
    for i, text in enumerate(texts):
        source += text
        splitnum += len(tokenize(text))
        if i < len(texts) - 1:
            source += ' ' + aspect + ' '
            splitnum += len(tokenize(aspect))
    if splitnum != len(tokenize(source.strip())):
        print(texts)
        print(aspect)
        print(source)
        print(splitnum)
        print(tokenize(source.strip()))
        print(len(tokenize(source.strip())))
        a = input('gfg')
    return re.sub(r' {2,}', ' ', source.strip())


def process_test(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    output = filename.replace('.raw', '.raw.order')

    with open(output, 'w', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(0, len(lines), 3)):
            texts = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].lower().strip()
            text = concat(texts, aspect)
            f.write(text + "\n")
            f.write(aspect + "\n")
            f.write(polarity + "\n")
            if i == len(lines) - 3:
                f.write(str(0))
            else:
                f.write(str(0) + "\n")


def process_local_test(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    output = filename.replace('.raw', '.raw.order.local')

    with open(output, 'w', encoding='utf-8') as f:
        for i in tqdm.tqdm(range(0, len(lines), 3)):
            texts = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].lower().strip()
            text = concat(texts, aspect)
            f.write(text + "\n")
            f.write(aspect + "\n")
            f.write(polarity + "\n")
            if i == len(lines) - 3:
                f.write(str(0))
            else:
                f.write(str(0) + "\n")


if __name__ == '__main__':
    process_train('./datasets/semeval14/restaurant_train.raw')
    process_test('./datasets/semeval14/restaurant_test.raw')
    process_train('./datasets/semeval14/laptop_train.raw')
    process_test('./datasets/semeval14/laptop_test.raw')
    process_train('./datasets/semeval15/restaurant_train.raw')
    process_test('./datasets/semeval15/restaurant_test.raw')
    process_train('./datasets/semeval16/restaurant_train.raw')
    process_test('./datasets/semeval16/restaurant_test.raw')
    process_train('./datasets/twitter/train.raw')
    process_test('./datasets/twitter/test.raw')
    # for i in range(10):
    #     process_local_train('./datasets/semeval14/restaurant_train.raw', i+1)
