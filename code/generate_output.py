from collections import defaultdict

lang_data = defaultdict(list)

with open('examples/test.txt') as f:
    for line in f.readlines():
        lang, lemma, inflection, tags = line.rstrip().split('\t')
        lang_data[lang].append((lemma, inflection, tags))
        
for k, v in lang_data.items():
    with open(f'guclasp-output/cl/{k}.test', '+w') as f:
        for a, b, c in sorted(v, key=lambda x: x[0]):
            f.write('\t'.join([a,b,c])+'\n')