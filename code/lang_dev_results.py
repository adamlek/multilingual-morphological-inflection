from collections import defaultdict
import numpy as np

lang_results =  defaultdict(list)
total = []

def read_file(path):
    with open(path) as f:
        for line in f.readlines():
            lang, lemma, pred, gold = line.rstrip().split('\t')
            lang_results[lang].append(pred == gold)
            total.append(pred == gold)
            #if lang == 'see':
            #    print(lemma, pred, gold)
            
    mean_lang = []
    for lang, res in sorted(lang_results.items(), key = lambda x: x[0]):
        print(lang, np.round(np.mean(res), 4))
        mean_lang.append(np.round(np.mean(res), 4))
    print('mean lang:', np.around(np.mean(mean_lang), 3))
    print('total:', np.round(np.mean(total), 4))
        
if __name__ == '__main__':
    read_file('examples/examples_dev_50.txt')
    read_file('examples/examples_dev_40.txt')