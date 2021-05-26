import os
from collections import defaultdict, Counter
from IPython import embed
from itertools import zip_longest, takewhile, chain
import toolz
from random import shuffle, choice, random
import torch
import pickle
from collections import namedtuple
from lev import levenshtein
from operator import add
import re

device = torch.device('cuda:0')
ops_vocab = {'<pad>':0, '<end>':1, '<start>':2, 'cp':3, 'add':4, 'sub':5, 'del':6}#, 'delsub':7, 'deladd':8, 'cpdel':9}

def numericalize(w, d):
    return [d.get(c, d.get('<unk>')) for c in w]

def add_specials(w, d):
    return [d['<start>']]+w+[d['<end>']]

def add_specials_opsx(w, d):
    return [d['<start>']]+w

def add_specials_gold(w, d, l):
    return l+w+[d['<end>']]

def add_specials_lang(w, d, l):
    return l+w

def pad_(b, pad_idx):
    max_len = max([len(x) for x in b])
    return [x+[pad_idx]*(max_len-len(x)) for x in b]
        
def moving_window(n, iterable):
  start, stop = 0, n
  while stop <= len(iterable):
      yield iterable[start:stop]
      start += 1
      stop += 1
 
def test_data_batcher(data, bs, char_vocab, tag_vocab, lang_vocab, cl=False):
    dlen = len(data)
    if not cl:
        shuffle(data)

    data = iter(data)
    Example = namedtuple('Example', ['lemmas', 'tags', 'langs', 'str_tags'])
    dataset = []
    
    for _ in range(int(dlen/bs)+1):
        batch = list(toolz.take(bs, data))
        if batch == []:
            return dataset
        
        shuffle(batch)

        lemmas, _, tags, langs, _ = zip(*batch)
        str_tags = tags
        
        langs = list(map(lambda x: numericalize(x.split(), lang_vocab), langs))
        lemmas = list(map(lambda x, y: add_specials_lang(numericalize(x, char_vocab), char_vocab, y), lemmas, langs))
        tags = list(map(lambda x: numericalize(x, tag_vocab), tags))
        
        lemmas = torch.tensor(pad_(lemmas, char_vocab['<pad>']), device=device) 
        tags = torch.tensor(pad_(tags, tag_vocab['<pad>']), device=device)
        langs = torch.tensor(langs, device=device)
        
        dataset.append(Example(lemmas, tags, langs, str_tags))
        
    return dataset

def lid_data_batcher(data, bs, char_vocab, tag_vocab, lang_vocab, cl=False):
    dlen = len(data)
    if not cl:
        shuffle(data)

    data = iter(data)
    Example = namedtuple('Example', ['lemmas', 'inflections', 'tags', 'ops_y', 'ops_x', 'langs'])
    dataset = []
    
    for _ in range(int(dlen/bs)+1):
        batch = list(toolz.take(bs, data))
        if batch == []:
            return dataset
        
        shuffle(batch)

        lemmas, inflections, tags, langs, ops = zip(*batch)
        
        ops_y = [list(filter(lambda m: m != 'del', map(lambda z: z.split('_')[0], x))) for x in ops]
        ops_x = [list(filter(lambda m: m != 'add', map(lambda z: z.split('_')[0], x))) for x in ops]
        
        langs = list(map(lambda x: numericalize(x.split(), lang_vocab), langs))
        lemmas = list(map(lambda x, y: add_specials_lang(numericalize(x, char_vocab), char_vocab, y), lemmas, langs))
        inflections = list(map(lambda x, y: add_specials_gold(numericalize(x, char_vocab), char_vocab, y), inflections, langs))
        
        ops_y = list(map(lambda x: add_specials(numericalize(x, ops_vocab), ops_vocab), ops_y))
        ops_x = list(map(lambda x: add_specials_opsx(numericalize(x, ops_vocab), ops_vocab), ops_x))
        tags = list(map(lambda x: numericalize(x, tag_vocab), tags))
        
        lemmas = torch.tensor(pad_(lemmas, char_vocab['<pad>']), device=device) 
        inflections = torch.tensor(pad_(inflections, char_vocab['<pad>']), device=device)
        tags = torch.tensor(pad_(tags, tag_vocab['<pad>']), device=device)
        langs = torch.tensor(langs, device=device)
        ops_y = torch.tensor(pad_(ops_y, ops_vocab['<pad>']), device=device)
        ops_x = torch.tensor(pad_(ops_x, ops_vocab['<pad>']), device=device)
        
        dataset.append(Example(lemmas, inflections, tags, ops_y, ops_x, langs))
    return dataset

def data_batcher(data, bs, char_vocab, tag_vocab, lang_vocab, cl=False):
    dlen = len(data)
    if not cl:
        shuffle(data)

    data = iter(data)
    Example = namedtuple('Example', ['lemmas', 'inflections', 'tags', 'langs', 'ops_y', 'ops_x'])
    dataset = []
    
    for _ in range(int(dlen/bs)+1):
        batch = list(toolz.take(bs, data))
        if batch == []:
            return dataset
        
        shuffle(batch)

        lemmas, inflections, tags, langs, ops = zip(*batch)
        
        ops_y = [list(filter(lambda m: m != 'del', map(lambda z: z.split('_')[0], x))) for x in ops]
        ops_x = [list(filter(lambda m: m != 'add', map(lambda z: z.split('_')[0], x))) for x in ops]
        
        ops_y = list(map(lambda x: add_specials(numericalize(x, ops_vocab), ops_vocab), ops_y))
        ops_x = list(map(lambda x: add_specials(numericalize(x, ops_vocab), ops_vocab), ops_x))
        lemmas = list(map(lambda x: add_specials(numericalize(x, char_vocab), char_vocab), lemmas))
        inflections = list(map(lambda x: add_specials(numericalize(x, char_vocab), char_vocab), inflections))
        tags = list(map(lambda x: numericalize(x, tag_vocab), tags))
        langs = list(map(lambda x: numericalize(x.split(), lang_vocab), langs))
        
        lemmas = torch.tensor(pad_(lemmas, char_vocab['<pad>']), device=device) 
        inflections = torch.tensor(pad_(inflections, char_vocab['<pad>']), device=device)
        tags = torch.tensor(pad_(tags, tag_vocab['<pad>']), device=device)
        langs = torch.tensor(langs, device=device)
        ops_y = torch.tensor(pad_(ops_y, ops_vocab['<pad>']), device=device)
        ops_x = torch.tensor(pad_(ops_x, ops_vocab['<pad>']), device=device)
        
        dataset.append(Example(lemmas, inflections, tags, langs, ops_y, ops_x))
    return dataset
            
if __name__ == '__main__':
    pass