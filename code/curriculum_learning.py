import numpy as np
from IPython import embed
from random import choices, shuffle, choice, randint, random
from collections import Counter, defaultdict
import math
import numpy as np
from data import get_batch, precompute_examples
from toolz import valfilter

class CurriculumSamplerDataset():
    def __init__(self, data, languages):
        self.data = data
        self.curriculum_data = []
        self.languages = languages
        self.cl_losses = {x:100.0 for x in range(len(data))}
        
    def most_copy_first(self):
        data = [self.find_cp_actions(x[4])/len(x[4]) for x in self.data]
        for i, d in enumerate(data):
            self.cl_losses[i] = -d*10
    
    def find_cp_actions(self, xs):
        return len([x for x in xs if x[:2] != 'cp'])
    
    def update_loss(self, scores, idxs):
        for idx, loss in zip(idxs, scores):
            self.cl_losses[idx] = loss
    
    def data_sampler(self, num_examples_per_lang=9000):
        uniform_lang_idxs = []

        all_idxs = [(i, x[3]) for i, x in enumerate(self.data)]
        
        for lang in self.languages.keys():
            # select all indexes belonging to the language
            lang_idxs = [x[0] for x in all_idxs if x[1] == lang]

            if len(lang_idxs) < 15000:
                num_examples_per_lang = len(lang_idxs)#randint(7000, len(lang_idxs))
            else:
                num_examples_per_lang = randint(7000, 15000)
            
            examples = []
            ids = []
            while len(examples) < num_examples_per_lang:
                example_idx = choice(lang_idxs)
                lang_idxs.remove(example_idx)
                #if example_idx not in ids:
                examples.append((self.cl_losses[example_idx], example_idx))
                #    ids.append(example_idx)
            uniform_lang_idxs += examples

        # reverse = false
        uniform_lang_data = [self.data[idx] for _, idx in sorted(uniform_lang_idxs, key=lambda x: x[0])]
        
        return uniform_lang_data, [x[0] for x in uniform_lang_idxs]

class CurriculumDataset():
    def __init__(self, data, languages):
        self.data = data
        self.curriculum_data = []
        self.languages = languages
        self.cl_losses = {x:0 for x in range(len(data))}
        
    def get_initial_scores(self, char_freq_vocab):
        initial_scores = []
        for i, example in enumerate(self.data):
            initial_scores.append((i, self.compute_unigram_logp(char_freq_vocab[example[3]], example)))
            
        initial_scores = sorted(initial_scores, key=lambda x: x[1], reverse=True)
        space = np.linspace(0, 1, len(initial_scores)).tolist()
        self.curriculum_scores = dict([(i, space[j]) for j, (i, _) in enumerate(initial_scores)])
        
    def compute_unigram_logp(self, char_freq_vocab, seq) -> float:
        """
        P(lemma)/T_lemma - P(inflection)/T_inflection
        """
        total = sum(char_freq_vocab.values())
        lemma = [char_freq_vocab[x]/total for x in seq[0]]
        inflection = [char_freq_vocab[x]/total for x in seq[0]]
        lemma = np.log(lemma).sum()/len(lemma)
        inflection = np.log(inflection).sum()/len(inflection)
        return lemma + inflection # (-) or *(+) ?
        
    def get_dataset(self):
        return self.curriculum_data
    
    def shuffle_dataset(self):
        shuffle(self.curriculum_data)
        
    def shortest_inflection_first(self):
        self.curriculum_data = self.data
        self.curriculum_data = sorted(self.curriculum_data, key=lambda x: len(x[1]), reverse=False)
        return self.curriculum_data

    def least_grammaticalfeatures_first(self):
        self.curriculum_data = self.data
        self.curriculum_data = sorted(self.curriculum_data, key=lambda x: len(x[2]), reverse=False)
        return self.curriculum_data
    
    def least_edits_per_feature_first(self):
        self.curriculum_data = self.data
        self.curriculum_data = sorted(self.curriculum_data, key=lambda x: (self.find_non_cp_actions(x[4])/len(x[2])), reverse=False)
        return self.curriculum_data
        
    def find_non_cp_actions(self, xs):
        return len([x for x in xs if x != 'cp'])
    
    def find_cp_actions(self, xs):
        return len([x for x in xs if x != 'cp'])
    
    def least_abs_diff_first(self):
        self.curriculum_data = self.data
        self.curriculum_data = sorted(self.curriculum_data, key=lambda x: abs(len(x[0])-len(x[1])), reverse=False)
        return self.curriculum_data
    
    def most_copy_first(self):
        self.curriculum_data = self.data
        # highest number first
        self.curriculum_data = sorted(self.curriculum_data, key=lambda x: (self.find_cp_actions(x[4])/len(x[4])), reverse=False)
        return self.curriculum_data
    
    def easyfirst_curriculum(self, scores=None):
        if scores is None:
            return self.random_curriculum()
        else:
            scores = np.argsort(scores)
            self.curriculum_data = [self.curriculum_data[x] for x in scores]
            return self.curriculum_data
        
    def loss_dataset(self, scores):
        scores = sorted(enumerate(scores), key=lambda x: x[1])
        return [(z[1], self.curriculum_data[x]) for z, x in zip(scores, range(len(self.curriculum_data)))]
        
    def random_curriculum(self):
        self.curriculum_data = self.data
        self.shuffle_dataset()
        return self.curriculum_data
        
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def sort_chunks_by_inflection_length(self, bin_size=128):
        new_curr = []
        for chunk in self.chunks(self.curriculum_data, bin_size):
            new_curr += sorted(chunk, key=lambda x: x[1], reverse=False)
        self.curriculum_data = new_curr
        
class CLLoader():
    def __init__(self, data, char_vocab, tag_vocab, lang_vocab):
        self.data = data#precompute_examples(data, char_vocab, tag_vocab, lang_vocab)
        self.languages = lang_vocab
        self.i_languages = {x:i for i, x in lang_vocab.items()}
        self.chars = char_vocab
        self.i_chars = {x:i for i, x in char_vocab.items()}
        self.tags = tag_vocab
        self.curriculum_scores = {x:1.0 for x in range(len(data))}
        self.initial_competence = 0.05
        self.competence = 0.05
        self.initial_prob = 1.0
        self.steps_until_fully_competent = 20000
        self.sort = False
        self.example_idxs = list(range(len(self.curriculum_scores)))
        
    def get_initial_scores(self, char_freq_vocab):
        initial_scores = []
        for i, example in enumerate(self.data):
            #embed()
            #lang_str = self.i_languages[example[5]]
            initial_scores.append((i, self.compute_unigram_logp(char_freq_vocab[example[3]], example)))
            
        initial_scores = sorted(initial_scores, key=lambda x: x[1], reverse=True)
        space = np.linspace(0, 1, len(initial_scores)).tolist()
        self.curriculum_scores = dict([(i, space[j]) for j, (i, _) in enumerate(initial_scores)])
        
    def compute_unigram_logp(self, char_freq_vocab, seq) -> float:
        """
        P(lemma)/T_lemma - P(inflection)/T_inflection
        """
        total = sum(char_freq_vocab.values())
        #lemma = [char_freq_vocab[x]/total for x in [self.i_chars[x] for x in seq[0][1:]]]
        #inflection = [char_freq_vocab[x]/total for x in [self.i_chars[x] for x in seq[0][1:]]]
        lemma = [char_freq_vocab[x]/total for x in seq[0]]
        inflection = [char_freq_vocab[x]/total for x in seq[0]]
        lemma = np.log(lemma).sum()/len(lemma)
        inflection = np.log(inflection).sum()/len(inflection)
        #embed()
        #assert False
        return lemma + inflection # (-) or *(+) ?
        
    def get_all_examples(self):
        return self.data
    
    def update_competence_linear(self):
        self.competence += 1/self.steps_until_fully_competent
        self.competence = np.clip(self.competence, 0.0, 1.0)
        
    def update_competence_sqrt(self, step):
        c02 = self.initial_competence**2
        nc = (1-c02)/self.steps_until_fully_competent
        competence = min(1,math.sqrt(step * nc + c02))
        self.competence = np.clip(competence, 0.0, 1.0)
        
    def update_competence_acc(self, acc):
        if acc > 0.7:
            self.competence += 1/self.steps_until_fully_competent
        self.competence = np.clip(self.competence, 0.0, 1.0)
    
    #
    def update_example_scores_norm(self, scores, idxs):
        upper = np.clip(self.competence*2, 0.0, 1.0)
        difficulty = np.linspace(0, upper, len(idxs)).tolist()
        scores = sorted(scores, reverse=True)
        for idx, _, diff in zip(idxs, scores, difficulty):
            self.curriculum_scores[idx] = diff
            
    def update_example_scores_loss(self, scores, idxs):
        upper = np.clip(self.competence*2, 0.0, 1.0)
        difficulty = np.linspace(0, upper, len(idxs)).tolist()
        scores = sorted(scores, reverse=False)
        for idx, _, diff in zip(idxs, scores, difficulty):
            self.curriculum_scores[idx] = diff
    
    def get_data(self, sort_by_score=True):
        if sort_by_score:
            examples_order = sorted(self.curriculum_scores.items(), key=lambda x: x[1])
        else:
            examples_order = list(self.curriculum_scores.items())
            shuffle(examples_order) 
            
        if self.competence < self.initial_competence:
            competence = self.initial_competence
        else:
            competence = self.competence
        return [(i, self.data[i]) for i, x in examples_order if x <= competence]
    
    def get_random_data(self):
        idxs = list(range(len(self.data)))
        shuffle(idxs)
        return [self.data[i] for i in idxs if random() <= self.initial_prob], idxs
    
    def sample_batch(self, batch_size):
        idxs = list(valfilter(lambda x: x <= self.competence, self.curriculum_scores).keys())
        batch_idxs = choices(idxs, k=batch_size)
        batch = get_batch([self.data[i] for i in batch_idxs], self.chars, self.tags, self.languages)
        batch_weights = [self.curriculum_scores[i] for i in batch_idxs]
        return batch, batch_weights, batch_idxs