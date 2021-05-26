import numpy as np
from IPython import embed
from random import shuffle, choice, randint
from collections import Counter

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
        
        #embed()
        #assert False
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
                
                
        