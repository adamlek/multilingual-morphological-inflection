import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler
import random

import os
from model import Inflector
from data import data_batcher, ops_vocab, test_data_batcher
import pickle
from IPython import embed
from pprint import pprint
from args import args
import curriculum_learning as cl
from madgrad import MADGRAD
from collections import Counter

#import tensorflow as tf
#import tensorboard as tb
#tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

device = torch.device('cuda:0')

def save_model(model, e):
    torch.save(model.state_dict(), f'./models/model_e={e}.pt')
    
def save_cs(cs, e):
    with open(f'losses/losses_{e}.txt', '+w') as f:
        for i, s in enumerate(cs):
            f.write('\t'.join([str(i), str(s)])+'\n')

def get_opt(model, lr=False):
    if args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr if lr else args.lr, weight_decay=args.wd)
    elif args.opt == 'madgrad':
        return MADGRAD(model.parameters(), lr=lr if lr else args.lr, weight_decay=args.wd)
    else:
        assert False
        
def orthogonal_initialization(model):
    for w in model.parameters():
        if len(w.size()) > 1:
            nn.init.orthogonal_(w.data)
            
def compute_lang_chars(train_data, langs):
    for lemma, infl, tags, lang, _ in train_data:
        langs[lang] += list(Counter(lemma + infl).keys())
    return {k:set(v) for k, v in langs.items()}

def compute_lang_filter_vecs(lang_chars, char_vocab, lang_vocab):
    smooth_vecs = {lang_vocab[k]:[] for k in lang_chars.keys()}
    for k, v in lang_chars.items():
        v = list(v) + ['<end>']
        smooth_vecs[lang_vocab[k]] = [int(x in v) for x in list(char_vocab.keys())]
    return torch.tensor(list(smooth_vecs.values()), device=device)
       
def languagewise_label_smoothing(gold_labels, filter_vecs, labels, mask, smoothing_value=0.05):
    to_one_hot = torch.eye(len(labels), device=device)
    gold_labels = to_one_hot[gold_labels]
    
    smoothed = (smoothing_value / filter_vecs.sum(1)).unsqueeze(1) # get smoothing value for each char in lang vocab
    smoothed = (smoothed * filter_vecs).unsqueeze(1).repeat(1, gold_labels.size(1), 1) # apply said value to one-hot vectors
    smoothed = (smoothed + (gold_labels*(1-smoothing_value))).view(-1, smoothed.size(-1)) # add value for the gold answer
    
    #mask = ~mask.bool() # invert mask matrix
    #smoothed = smoothed.masked_fill_(mask.view(-1,1).repeat(1,len(labels)), 0) # ignore masked tokens
    smoothed = smoothed.view(gold_labels.size(0), gold_labels.size(1), len(labels))
    
    return smoothed

def load_and_test_model():
    
    with open('data_pkl/data_wHallucinations_wOps.pkl', 'rb') as f:
        _, dev_data, lf, cf, tf, char_vocab, tag_vocab, lang_vocab = pickle.load(f)
        
    model = Inflector(char_vocab['<start>'], 
                      char_vocab['<end>'],
                      (char_vocab['<pad>'], tag_vocab['<pad>'], ops_vocab['<pad>']),
                      len(char_vocab.keys()), 
                      len(tag_vocab.keys()), 
                      len(lang_vocab.keys()),
                      len(ops_vocab.keys()))
    model.to(device)
    model.load_state_dict(torch.load(f'models/model_e={args.n_epochs-1}.pt'))
    
    # Test
    test_iter = data_batcher(dev_data, args.bs, char_vocab, tag_vocab, lang_vocab)
    em_test, best_b = test(0, model, char_vocab, test_iter)
    print('--- TEST ---')
    print('EM beam search:', np.round(np.mean(em_test), 3))
    print('best beams:', Counter(best_b))
    print()
    
    
def main():
    #print(args)
    
    #if args.hallucinated_examples == 10000:
    #    data_path = 'data_pkl/data_w-10kHallucinations_wOps.pkl'
    #else:
    
    data_path = 'data_pkl/data_test_release-20khall.pkl'
        
    with open(data_path, 'rb') as f:
        train_data, dev_data, test_data, lf, cf, tf, char_vocab, tag_vocab, lang_vocab = pickle.load(f)
    
    print('train dataset size:', len(train_data), len(train_data)/args.bs)
    print('dev dataset size:', len(dev_data))
    print('test dataset size:', len(test_data))
    print('---')
    print('Building model...')
    
    lang_chars = compute_lang_chars(train_data, {lang:[] for lang in lang_vocab})
    lang_chars = compute_lang_filter_vecs(lang_chars, char_vocab, lang_vocab)

    model = Inflector(char_vocab['<start>'], 
                      char_vocab['<end>'],
                      (char_vocab['<pad>'], tag_vocab['<pad>'], ops_vocab['<pad>']),
                      len(char_vocab.keys()), 
                      len(tag_vocab.keys()), 
                      len(lang_vocab.keys()),
                      len(ops_vocab.keys()))
    model.to(device)
    if args.model_orthogonal_init:
        orthogonal_initialization(model)
    #print(model)
    
    proposed_path = f'runs/inflection-opt={args.opt}-lr{args.lr}-wd{args.wd}-bs{args.bs}-d{args.hidden_dim}-orth={args.model_orthogonal_init}-curr={args.use_curriculum}-s={args.do_smoothing}-cs={args.initial_curriculum}-cos_att={args.char_att_cos}'
    while os.path.isdir(proposed_path):
        proposed_path += 'x'
    writer = SummaryWriter(proposed_path)
    
    optimizer = get_opt(model)
    #num_steps = int(len(train_data)/args.bs)*args.n_epochs
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=args.min_lr)
    #print(num_steps)
    
    #if args.use_curriculum:
    print('Constructing curriculum...')
    data_class = cl.CurriculumDataset(train_data)
    
    if args.initial_curriculum == 0:
        train_data = data_class.shortest_inflection_first()
    elif args.initial_curriculum == 1:
        train_data = data_class.most_copy_first()
    elif args.initial_curriculum == 2:
        train_data = data_class.least_grammaticalfeatures_first()
    elif args.initial_curriculum == 3:
        train_data = data_class.least_edits_per_feature_first()
    elif args.initial_curriculum == 4:
        train_data = data_class.least_abs_diff_first()
    else:
        train_data = data_class.random_curriculum()
    
    dev_iter = data_batcher(dev_data, args.bs, char_vocab, tag_vocab, lang_vocab)
    
    print('Starting training...')
    for e in range(args.n_epochs):
        train_iter = data_batcher(train_data, args.bs, char_vocab, tag_vocab, lang_vocab, cl=args.use_curriculum)
        #print('iter len', sum([len(x[0]) for x in train_iter]), sum([len(x[0]) for x in dev_iter]))
        num_steps = len(train_iter)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=args.min_lr)
        
        # Train
        model, curriculum_scores = train(e+1, model, char_vocab, train_iter, writer, optimizer, lang_chars, lr_scheduler)
        # Sort train examples by loss
        train_data = data_class.easyfirst_curriculum(curriculum_scores)
        
        # save stuff
        save_model(model, e)
        save_cs(curriculum_scores, e)
        
        if not args.sp_step_decay:
            # update scheduled sampling rate every epoch
            model.sc_e -= args.prob_decay_epoch if model.sc_e > 0 else 0
        
        # Validate
        dev_outputs = validate(e+1, model, char_vocab, dev_iter, writer, lf, lang_vocab)
        
    test_iter = test_data_batcher(test_data, args.bs, char_vocab, tag_vocab, lang_vocab)
    test(0, model, char_vocab, lang_vocab, test_iter)
    
    # print latex table
    #format_outputs(dev_outputs)
        
def train(e, model, char_vocab, data_iter, writer, optimizer, lang_chars, lr_scheduler=None, warmup_scheduler=None):
    model.train()
    curriculum_scores = []
    
    torch.cuda.empty_cache() # just cleanup...
    
    for ex_i, example in enumerate(data_iter):
        print(ex_i, end='\r')
        optimizer.zero_grad()
        lemma_mask = (example.lemmas != model.c_pad).int()
        tag_mask = (example.tags != model.t_pad).int()
        target_mask = (example.inflections[:,1:] != model.c_pad).int()
        
        decode_pred, lang_pred, opsy_pred, opsx_pred = model(example.lemmas,
                                                             lemma_mask,
                                                             example.tags,
                                                             tag_mask,
                                                             example.langs,
                                                             example.inflections,
                                                             test=False)


        if args.do_smoothing:
            with torch.no_grad():
                filter_vecs = lang_chars[example.langs].squeeze(1)
                smoothed_gold_labels = languagewise_label_smoothing(example.inflections[:,1:], 
                                                                    filter_vecs, 
                                                                    char_vocab, 
                                                                    target_mask, 
                                                                    args.smoothing_value)
            decode_loss = _kl_div_loss(F.log_softmax(decode_pred, -1), smoothed_gold_labels)
            gold = torch.argmax(smoothed_gold_labels, 1)
            #decode_loss = _cross_entropy_with_probs(decode_pred.view(-1, decode_pred.size(-1)), smoothed_gold_labels, cross_entropy_with_probs, model.c_pad)
        else:
            decode_loss = _loss(decode_pred, example.inflections[:,1:], F.cross_entropy, model.c_pad)
            gold = example.inflections[:,1:]
        
        opsy_loss = _loss(opsy_pred, example.ops_y[:,1:], F.cross_entropy, model.o_pad)
        opsx_loss = _loss(opsx_pred, example.ops_x, F.cross_entropy, model.o_pad)
        decode_ex_acc = _ex_acc(decode_pred, example.inflections[:,1:]) 
        decode_acc = _acc(decode_pred, example.inflections[:,1:])
        lang_loss = _loss(lang_pred, example.langs, F.cross_entropy)
        lang_acc = _acc(lang_pred, example.langs)
        
        loss = decode_loss + lang_loss + opsy_loss + opsx_loss
        loss.backward()
        
        # clip grads after gradient calculation
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if warmup_scheduler is not None:
            warmup_scheduler.dampen()
            
        non_zero_idx = [len(torch.nonzero(x)) for j, x in enumerate(gold)]
        
        curriculum_scores += [_loss(decode_pred[m,:], 
                                    example.inflections[m,1:], 
                                    F.cross_entropy, 
                                    model.c_pad).item() 
                              for m in range(gold.size(0))]
        #print(ex_i, len(curriculum_scores))

        ex_accs = [int(sum(x[:j])==j) for x, j in zip(decode_ex_acc, non_zero_idx)]
        
        #cuda_time = example.inflections.size(1)/prof.key_averages().total_average().cuda_time

        writer.add_scalar('Ops X train loss', opsy_loss.item(), args.c)
        writer.add_scalar('Ops Y train loss', opsx_loss.item(), args.c)
        writer.add_scalar('Decode train loss', decode_loss.item(), args.c)
        writer.add_scalar('Decode train acc', np.mean(decode_acc), args.c)
        #writer.add_scalar('Levenshtein train dist', np.mean(lev_d_history), args.c)
        writer.add_scalar('Decode ex train acc', np.mean(ex_accs), args.c)
        writer.add_scalar('Language train loss', np.mean(lang_loss.item()), args.c)
        writer.add_scalar('Language train acc', np.mean(lang_acc), args.c)
        writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], args.c)
        
        #writer.add_scalar('Sampling P', model.sc_e, args.c)
        args.c += 1
        
        #if ex_i >= 10:
        #    return model, curriculum_scores
        
    return model, curriculum_scores

def validate(e, model, char_vocab, data_iter, writer, dev_languages, lang_vocab):
    outputs = {lang:{'lev':[], 
                     'string_acc':[],
                     'string_nonpad_acc':[],
                     'em_acc':[],
                     'lang_id_acc':[], 
                     'num_examples':sizes['dev'], 
                     'output_pairs':[],
                     'with_hallucinated':True} for lang, sizes in dev_languages.items()}
    
    itos = {v:k for k, v in char_vocab.items()}
    lang_vocab_ = {i:x for x, i in lang_vocab.items()}
    out_file = open(f'examples/examples_dev_{e}.txt', '+w')    
    
    model.eval()
    d_loss_history = []
    d_acc_history = []
    d_ex_acc_history = []
    l_loss_history = []
    l_acc_history = []
    em_acc_history = []
    
    for i, example in enumerate(data_iter):
        #print(i)
        lemma_mask = (example.lemmas != 0).int()
        tag_mask = (example.tags != 0).int()
        with torch.no_grad():
            decode_pred, lang_pred, _, _ = model(example.lemmas,
                                                 lemma_mask,
                                                 example.tags,
                                                 tag_mask,
                                                 example.langs,
                                                 example.inflections,
                                                 tf=False,
                                                 test=False)
            
        decode_loss = _loss(decode_pred, example.inflections[:,1:], F.cross_entropy, model.c_pad)
        decode_ex_acc, decode_acc = _ex_acc(decode_pred, example.inflections[:,1:]), _acc(decode_pred, example.inflections[:,1:])
        lang_loss, lang_acc = _loss(lang_pred, example.langs, F.cross_entropy), _acc(lang_pred, example.langs)
        
        #gold = torch.argmax(example.inflections[:,1:], -1)
        preds = torch.argmax(decode_pred, -1)
        for j in range(example.langs.size(0)):
            lang_j = lang_vocab_[example.langs[j].item()]
            gold_j = example.inflections[j,1:]
            input_j = example.lemmas[j,1:]
            pred_j = preds[j]
            non_pad_idxs = len(torch.nonzero(gold_j).squeeze(-1))
            
            string_acc = (gold_j == pred_j).int().tolist()
            string_nonpad_acc = (gold_j[:non_pad_idxs] == pred_j[:non_pad_idxs]).int().tolist()
            decoded_output = ''.join([itos[x] for x in pred_j.tolist() if x not in [0,1,2]])
            decoded_input = ''.join([itos[x] for x in input_j.tolist() if x not in [0,1,2]])
            decoded_gold = ''.join([itos[x] for x in gold_j.tolist() if x not in [0,1,2]])
            em_acc = int(decoded_gold == decoded_output)
            em_acc_history.append(em_acc)
            
            outputs[lang_j]['string_acc'] += string_acc
            outputs[lang_j]['string_nonpad_acc'] += string_nonpad_acc
            outputs[lang_j]['em_acc'].append(em_acc)
            outputs[lang_j]['output_pairs'].append((decoded_input, decoded_gold, decoded_output))
            outputs[lang_j]['lang_id_acc'] += [lang_acc[j]]
            
            if decoded_output != decoded_gold:
                out_file.write('\t'.join([str(e), decoded_input, decoded_output, decoded_gold, lang_j]) + '\n')
    
        d_loss_history.append(decode_loss.item())
        d_acc_history += decode_acc
        l_loss_history.append(lang_loss.item())
        l_acc_history += lang_acc
        
    writer.add_scalar('Dev EM', np.mean(em_acc_history), e)
    writer.add_scalar('Dev acc', np.mean(d_acc_history), e)

    print('DEV', e,
        '|',
        np.round(np.mean(d_loss_history), 4), 
        np.round(np.mean(d_acc_history), 4),
        np.round(np.mean(em_acc_history), 4))
    #print()
    #format_outputs(outputs)
    return outputs

def test(e, model, char_vocab, lang_vocab, data_iter):    
    itos = {v:k for k, v in char_vocab.items()}   
    lang_vocab_ = {i:x for x, i in lang_vocab.items()}
    
    test_out = open('examples/test.txt', '+w')
    model.eval()
    em_acc = []
    best_b = []
    for i, example in enumerate(data_iter):
        
        lemma_mask = (example.lemmas != 0).int()
        tag_mask = (example.tags != 0).int()
        with torch.no_grad():
            decoded_batch, b = model(example.lemmas,
                                     lemma_mask,
                                     example.tags,
                                     tag_mask, 
                                     test=True)
            
            best_b += b
            
        for j, output in enumerate(decoded_batch):

            lang_j = lang_vocab_[example.langs[j].item()]
            prediction_j = ''.join([itos[x] for x in output[0] if x not in [0, 1, 2]])
            lemma_j = ''.join([itos[x.item()] for x in example.lemmas[j,:] if x not in [0, 1, 2]])
            test_out.write('\t'.join([lang_j, lemma_j, prediction_j, ';'.join(example.str_tags[j])])+'\n')
    
    test_out.close()
    return em_acc, best_b

def _cross_entropy_with_probs(pred, gold, loss_fn, ignore_idx):
    return loss_fn(pred, gold, ignore_idx=ignore_idx)

def _kl_div_loss(pred, gold):
    # mean
    return F.kl_div(pred, gold, reduction='batchmean')

def _kl_div_loss2(pred, gold):
    return torch.mean(-(gold * pred).sum(-1), -1)

def _loss(pred, gold, loss_fn, ignore_idx=False):
    if ignore_idx:
        return loss_fn(pred.view(-1,pred.size(-1)), gold.contiguous().view(-1), ignore_index=ignore_idx)
    else:
        return loss_fn(pred.view(-1,pred.size(-1)), gold.contiguous().view(-1))

def _acc(pred, gold):
    return (torch.argmax(pred.view(-1,pred.size(-1)),1) == gold.contiguous().view(-1)).int().tolist()

def _ex_acc(pred, gold):
    return (torch.argmax(pred, -1) == gold).int().tolist()

def format_outputs(outputs, show_all=True):
    
    with open('latest_out.pkl', '+wb') as f:
        pickle.dump(outputs, f)
            
    total_em = []
    total_str_acc = []
    for lang, scores in sorted(outputs.items(), key=lambda x: x[0]):
        if show_all:
            pass
        else:
            if not scores['with_hallucinated']:
                continue
        total_em += scores['em_acc']
        total_str_acc += scores['string_nonpad_acc']
        print(' & '.join(list(map(lambda x: str(x), [lang,
                                                  np.round(np.mean(scores['string_acc']), 3), 
                                                  np.round(np.mean(scores['string_nonpad_acc']), 3),
                                                  np.round(np.mean(scores['em_acc']), 3)])))+' \\\\')
        
        print(' & ').join(list(list(map(lambda x: str(x), ['Mean', 
                                                           '_', 
                                                           np.mean(total_str_acc, 3), 
                                                           np.mean(total_em,3)]))))
        #print('---')


def test_hp():
    for ed in [64,256]:
        args.embedding_dim = ed
        for hd in [256, 512]:
            args.hidden_dim = hd
            args.c = 0
            print('--->', ed, hd)
            main()
    
if __name__ == '__main__':

    main()
    #load_and_test_model()
        
    #test_hp()