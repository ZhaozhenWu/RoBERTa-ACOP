# -*- coding: utf-8 -*-
import itertools
import re

import spacy
import tqdm

nlp = spacy.load('en_core_web_sm')


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
                if index == 0:
                    f.write(str(1) + "\n")
                elif i == len(lines) - 3 and index == 5:
                    f.write(str(0))
                else:
                    f.write(str(0) + "\n")


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


if __name__ == '__main__':
    process_train('./datasets/semeval14/restaurant_train.raw')
    process_test('./datasets/semeval14/restaurant_test.raw')
    process_train('./datasets/semeval14/laptop_train.raw')
    process_test('./datasets/semeval14/laptop_test.raw')
    process_train('./datasets/semeval15/restaurant_train.raw')
    process_test('./datasets/semeval15/restaurant_test.raw')
    process_train('./datasets/semeval16/restaurant_train.raw')
    process_test('./datasets/semeval16/restaurant_test.raw')
    process_train('./datasets/acl-14-short-data/train.raw')
    process_test('./datasets/acl-14-short-data/test.raw')