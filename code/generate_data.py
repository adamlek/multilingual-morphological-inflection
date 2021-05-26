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
import string
from Bio import pairwise2

def get_ngrams(seq, max_n):
    grams = []
    for n in range(1,max_n+1):
        grams += list(map(lambda x: ''.join(x), zip(*[seq[i:] for i in range(n)])))
    return set(grams)

def read_test(path):
    dataset = []
    with open(path) as p:
        for line in p:
            lemma, tags, lang = line.rstrip().split('\t') + [path.split('/')[-1].split('.')[0]]
            dataset.append([lemma, '_', tags.split(';'), lang, '_'])
    return dataset
            
def read_file(path, char_freq=None, tag_freq=None):
    dataset = []
    char_grams = set()
    with open(path) as p:
        for line in p:
            lemma, inflection, tags, lang = line.rstrip().split('\t') + [path.split('/')[-1].split('.')[0]]
            char_grams = char_grams.union(get_ngrams(lemma, 3))
            tags = tags.split(';')
            operations = oracle_actions(string_alignment2(lemma, inflection), lemma, inflection)
            dataset.append([lemma, inflection, tags, lang, operations])
            if char_freq is not None:
                for w in [lemma, inflection]:
                    char_freq = toolz.merge_with(sum, char_freq, Counter(list(w)))
                tag_freq = toolz.merge_with(sum, tag_freq, Counter(tags))
    return dataset, char_freq, tag_freq, char_grams

def compute_lang_vocab(data):
    cv = defaultdict(int)
    for lemma, infl, *_ in data:
        for c in lemma+infl:
            cv[c] += 1
    return cv

def hallucinate_datafiles(num_hall=10000):
    char_freq, tag_freq = {'<pad>':0, '<end>':0, '<start>':0, '<unk>':0}, {'<pad>':0, '<unk>':0}
    pref = '../test_data_release/all_languages/'
    for lang in os.listdir('../test_data_release/all_languages/'):
        if lang.endswith('train'):
            data, char_freq, tag_freq, n_grams = read_file(f'{pref}{lang}', char_freq, tag_freq)
            if len(data) < 10000:
                print('hallucinating on language:', lang)
                hallucinated_data = get_hallucinated_data(data, n_grams, num_hall)
                save_new_data(hallucinated_data, lang, pref)
        else:
            continue

def save_new_data(data, filename, path):
    filename = path+filename.split('.')[0]+'.hall'
    with open(filename, '+w') as f:
        for lemma, infl, tags, lang, ops in data:
            tags = ';'.join(tags)
            f.write('\t'.join([lemma, infl, tags])+'\n')

def get_data():
    train, dev, test = [], [], []
    char_freq, tag_freq = {'<pad>':0, '<end>':0, '<start>':0, '<unk>':0}, {'<pad>':0, '<unk>':0}
    lang_data = defaultdict(dict)
    pref = '../test_data_release/all_languages/'
    for lang in os.listdir('../test_data_release/all_languages/'):
        
        if lang.endswith('train') or lang.endswith('hall'):
            data, char_freq, tag_freq, n_grams = read_file(f'{pref}{lang}', char_freq, tag_freq)
            train += data
        elif lang.endswith('test'):
            data = read_test(f'{pref}{lang}')
            print(lang, len(data))
            test += data
        else:
            data, *_ = read_file(f'{pref}{lang}')
            dev += data
        lang, dtype = lang.split('.')
        lang_data[lang][dtype] = len(data)
            
    return train, dev, test, lang_data, char_freq, tag_freq

def oracle_actions(editops, x, y):
    actions = []
    for action in editops:
        if action['type'] == 'copy':
            lc, ic = x[action['i']], y[action['j']]
            actions.append(f'cp_{lc}')
        elif action['type'] == 'insertion':
            ic = y[action['j']]
            actions.append(f'add_{ic}')
        elif action['type'] == 'substitution':
            lc, ic = x[action['i']], y[action['j']]
            actions.append(f'sub_{lc}_{ic}')
        elif action['type'] == 'deletion':
            lc = x[action['i']]
            actions.append(f'del_{lc}')
        else:
            assert False
    return actions

def string_alignment2(lemma, inflected):
    ops = levenshtein(lemma, inflected)[1]
    return ops

def al(lemma, inflected):
    ops = levenshtein(lemma, inflected)
    return ops

def get_hallucinated_data(data, vocab, num_examples):
    new_examples = []
    #data, vocab = get_lang_stats(f'../development_languages/{lang}.train')
    
    while len(new_examples) < num_examples:
        x, y, t, l, ops = choice(data)
        entries = compute_example_with_ngrams(x, y, t, l, vocab)
        #compute_example_test_patterns(x, y, t, l, vocab)
        entries = [x for x in entries if x not in new_examples]
        if entries:
            # old (submission)
            #new_entries = entries[:2] + [x for x in entries[2:] if random() > 0.5]
            #if new_entries:
            new_examples += [choice(entries)]#new_entries
        #print(len(new_examples), end='\r')
    return new_examples

def ngram_selector(total, n_grams=[1,2,3]):
    cur = []
    while sum(cur) != total:
        c = choice(n_grams)
        if sum(cur + [c]) <= total:
            cur.append(c)
    return cur

def compute_example_with_ngrams(x, y, t, l, ngrams):
    entries = []
    y = ''.join([c for c in y])
    xs = list(chain(*[list(toolz.accumulate(add, x[i:])) for i in range(len(x))]))
    xs = sorted(list(filter(lambda z: len(z)>=3, xs)), reverse=True, key=len)
    replacements = [c for c in xs if c in y]
    
    if x == y:
        replacements = list(filter(lambda k: len(k) < len(x)-1, replacements))
        
    shuffle(replacements)

    for s in replacements:
        yb, ye = re.search(s, y).span()
        yb += 1
        ye -= 1
        xb, xe = re.search(s, x).span()
        xb += 1
        xe -= 1
        
        # only take middle pat of selected substring
        s = s[1:-1]
        # dont replace phonological indicators
        if s in ["'", '-', 'ː', ':', "`", 'ˈ', '(', ')', '{', '}', '*', '=']:
        #if s in ["'", '-', 'ː', ':', "`"]:
            continue
                
        ny = [c for c in y]
        nx = [c for c in x]
        ny = set_range_in_string_to_symb(yb, ye, ny)
        nx = set_range_in_string_to_symb(xb, xe, nx)
        
        replace_parts = []
        chars_to_add = list(filter(lambda x: x >= 1, [len(s)-2, len(s)-1, len(s), len(s)+1, len(s)+2]))
        chars_to_add = choice(chars_to_add)

        for n in ngram_selector(chars_to_add):
            ngram_subset = list(filter(lambda x: len(x) == n, ngrams))
            replace_parts.append(choice(ngram_subset))
            
        replace_parts = ''.join(replace_parts)
        ny = ''.join(set_range_in_string_to_seq2(yb, ye, ny, replace_parts))
        nx = ''.join(set_range_in_string_to_seq2(xb, xe, nx, replace_parts))
        ops = oracle_actions(string_alignment2(nx, ny), nx, ny)
        
        if nx != x and ny != y:
            entries.append([nx, ny, t, l, ops])

    return entries

def compute_example_test_patterns(x, y, t, l, vocab):
    entries = []
    _y = ''.join([c for c in y])
    xs = list(chain(*[list(toolz.accumulate(add, x[i:])) for i in range(len(x))]))
    xs = sorted(list(filter(lambda z: len(z)>=3, xs)), reverse=True, key=len)
    
    for item in xs:
        try:
            #c = 0
            yb, ye = re.search(item, y).span()
            xb, xe = re.search(item, x).span()
            
            new_y2 = [c for c in y]
            new_x2 = [c for c in x]
            
            new_y2 = set_range_in_string_to_symb(yb, ye, new_y2)
            new_x2 = set_range_in_string_to_symb(xb, xe, new_x2)
                
            replace_part = []
            for k, c in enumerate(item):
                if random() < 0.8:
                    new_char = choice(list(filter(lambda x: x not in string.punctuation, vocab.keys())))
                else:
                    new_char = c
                replace_part.append(new_char)
            
            new_y2 = ''.join(set_range_in_string_to_seq(yb, ye, new_y2, replace_part))
            new_x2 = ''.join(set_range_in_string_to_seq(xb, xe, new_x2, replace_part))
            ops = oracle_actions(string_alignment2(new_x2, new_y2), new_x2, new_y2)
            entries.append([new_x2, new_y2, t, l, ops])
        except Exception as e:
            #print(e)
            continue
    return entries
    
def set_range_in_string_to_seq2(s, e, string, seq):
    if not isinstance(string, list):
        string = list(string)
    
    s = ''.join(string[:s])
    e = ''.join(string[e:])
    new = ''.join([s, seq, e])
    
    return new
    
def set_range_in_string_to_seq(s, e, string, seq):
    if not isinstance(string, list):
        string = list(string)
    for j, i in enumerate(range(s, e)):
        string[i] = seq[j]
    return ''.join(string)
        
def set_range_in_string_to_symb(s, e, string, symb='-'):
    if not isinstance(string, list):
        string = list(string)

    for i in range(s, e):
        string[i] = symb
    
    return ''.join(string) 

def save_data_as_pkl():
    train, dev, test, lf, cf, tf = get_data()
    char_vocab = {x:i for i, x in enumerate(cf.keys())}
    tag_vocab = {x:i for i, x in enumerate(tf.keys())}
    lang_vocab = {x:i for i, x in enumerate(lf.keys())}

    with open('data_pkl/data_test_release-10khall-special_symbs.pkl', '+wb') as f:
        pickle.dump((train, dev, test, lf, cf, tf, char_vocab, tag_vocab, lang_vocab), f)
        
if __name__ == '__main__':
    #hallucinate_datafiles()
    save_data_as_pkl()