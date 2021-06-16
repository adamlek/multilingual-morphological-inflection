import numpy as np
from IPython import embed
from random import choices, shuffle, choice, randint, random
from collections import Counter, defaultdict
import math
import numpy as np
from data import get_batch, precompute_examples
from toolz import valfilter
from args import args

class CLLoaderBatchSampling():
    def __init__(self, data, char_vocab, tag_vocab, lang_vocab):
        self.data = precompute_examples(data, char_vocab, tag_vocab, lang_vocab)
        self.languages = lang_vocab
        self.i_languages = {x:i for i, x in lang_vocab.items()}
        self.chars = char_vocab
        self.i_chars = {x:i for i, x in char_vocab.items()}
        self.tags = tag_vocab
        self.curriculum_scores = [[i, 0.999] for i in range(len(data))]
        self.initial_competence = 0.05
        self.competence = 0.05
        self.steps_until_fully_competent = args.full_competence
        self.example_idxs = list(range(len(self.curriculum_scores)))
        self.times_not_seen = np.array([5 for _ in range(len(self.data))])
        self.example_language_weights = self.compute_lang_weights()
        
    def compute_lang_weights(self):
        lang_freq = Counter([self.data[i][0][0] for i in self.example_idxs])
        lang_freq = np.array(list(lang_freq.values()))/sum(lang_freq.values())
        lang_freq = np.reciprocal(lang_freq)
        lang_freq = lang_freq/lang_freq.sum()
        lang_freq = [lang_freq[self.data[i][0][0]] for i in self.example_idxs]
        return lang_freq
        
    def get_initial_scores(self, char_freq_vocab):
        initial_scores = []
        for i, example in enumerate(self.data):
            lang = self.i_languages[example[5]]
            initial_scores.append((i, self.compute_unigram_logp(char_freq_vocab[lang], example)))
            
        initial_scores = sorted(initial_scores, key=lambda x: x[1], reverse=False)
        space = np.linspace(0, 1, len(initial_scores)).tolist()
        # [data_idx, score]
        for j, (idx, _) in enumerate(initial_scores):
            self.curriculum_scores[idx] = [idx, space[j]]
        #self.curriculum_scores = [[i, space[j]] for j, (i, _) in enumerate(initial_scores)]
        
        
    def compute_unigram_logp(self, char_freq_vocab, seq) -> float:
        """
        P(lemma)/T_lemma - P(inflection)/T_inflection
        """
        total = sum(char_freq_vocab.values())
        lemma = [char_freq_vocab[x]/total for x in [self.i_chars[x] for x in seq[0][1:]]]
        inflection = [char_freq_vocab[x]/total for x in [self.i_chars[x] for x in seq[0][1:]]]
        #denom = len(lemma) + len(inflection)
        lemma = np.log(lemma).sum()#/len(lemma)
        inflection = np.log(inflection).sum()#/len(inflection)
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
    
    def update_example_scores_norm(self, scores, idxs):
        difficulty = np.linspace(0, 1, len(idxs)).tolist()
        scores = sorted(scores, reverse=True)
        for (idx, _), _, diff in zip(idxs, scores, difficulty):
            self.curriculum_scores[idx][1] = diff
            
    def update_example_scores_loss(self, scores, idxs):
        #lower = np.clip(self.competence, 0, 1)# - 0.3, 0, 1)
        #upper = np.clip(self.competence + 0.1, 0, 1) # +0.2?
        difficulty = np.linspace(0, 1, len(idxs)).tolist() 
        scores = sorted(scores, reverse=False)
        for (idx, _), _, diff in zip(idxs, scores, difficulty):
            self.curriculum_scores[idx][1] = diff
            
    def redistribute_probabilities(self):
        redistrib_scores = sorted(self.curriculum_scores, key=lambda x: x[1])
        space = np.linspace(0, 1, len(redistrib_scores)).tolist()
        for j, (idx, _) in enumerate(redistrib_scores):
            self.curriculum_scores[idx] = [idx, space[j]]
            
    def sample_with_weight(self, batch_size):
        batch_idxs = []
        while len(batch_idxs) != batch_size:
            # (data_idx, score)
            i, s = choice(self.curriculum_scores)
            if random() < self.example_language_weights[i] and (i, s) not in batch_idxs:
                batch_idxs.append((i, s))
        
        batch = get_batch([self.data[i] for i, _ in batch_idxs], self.chars, self.tags, self.languages)
        batch_weights = [s for _, s in batch_idxs]
        
        return batch, batch_weights, batch_idxs
    
    def sample_batch(self, batch_size):
        batch_idxs = []
        steps = 0
        
        while len(batch_idxs) != batch_size:
            # (data_idx, score)
            i, s = choice(self.curriculum_scores)
            # (self.competence-0.2) <= 
            if s <= self.competence and (i,s) not in batch_idxs:
                batch_idxs.append((i,s))
            steps += 1
            if steps == 50000:
                print('> redistributing probabilities')
                self.redistribute_probabilities()
                steps = 0
        
        batch = get_batch([self.data[i] for i, _ in batch_idxs], self.chars, self.tags, self.languages)
        batch_weights = [s/self.competence for _, s in batch_idxs]
        
        return batch, batch_weights, batch_idxs