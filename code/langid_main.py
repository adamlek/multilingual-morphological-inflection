from re import A
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
from toolz import take
from lev import eval_form

import os
from langid_model import Inflector
from data import lid_data_batcher, ops_vocab, test_data_batcher
import pickle
from IPython import embed
from pprint import pprint
from args import args
import curriculum_learning_2 as cl
from madgrad import MADGRAD
from collections import Counter, defaultdict
import cProfile

device = torch.device('cuda:0')

def save_model(model):
    torch.save(model.state_dict(), f'./models/batch-sampler-lid_model_params={args.curriculum_learning}_{args.label_smoothing}_{args.multitask_learning}_{args.scheduled_sampling}_{args.model_orthogonal_init}_{args.mode}.pt')
    
def save_cs(cs, e):
    with open(f'losses/lid_losses_{e}.txt', '+w') as f:
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
    
    # mask = ~mask.bool() # invert mask matrix
    # smoothed = smoothed.masked_fill_(mask.view(-1,1).repeat(1,len(labels)), 0) # ignore masked tokens
    smoothed = smoothed.view(gold_labels.size(0), gold_labels.size(1), len(labels))
    
    return smoothed

def dev_to_test_format(ex):
    ex[1] = '_'
    ex[-1] = '_'
    return ex

def load_and_test_model():
    with open('data_pkl/data_test_release-10khall-special_symbs.pkl', 'rb') as f:
        _, dev_data, test_data, lf, cf, tf, char_vocab, tag_vocab, lang_vocab = pickle.load(f)

    model = Inflector(char_vocab['<start>'], 
                      char_vocab['<end>'],
                      (char_vocab['<pad>'], tag_vocab['<pad>'], ops_vocab['<pad>']),
                      len(char_vocab.keys()), 
                      len(tag_vocab.keys()), 
                      len(lang_vocab.keys()),
                      len(ops_vocab.keys()))
    model.to(device)
    model.load_state_dict(torch.load('models/final_model.pt'))

    dev_iter = lid_data_batcher(dev_data, args.bs, char_vocab, tag_vocab, lang_vocab)
    validate(0, model, char_vocab, dev_iter, False, lf, lang_vocab)

    #gold = [x[1] for x in dev_data]
    #dev_data = [dev_to_test_format(x) for x in dev_data]

    #test_iter = test_data_batcher(dev_data, args.bs, char_vocab, tag_vocab, lang_vocab)
    #test(0, model, char_vocab, lang_vocab, test_iter, gold)
    
def compute_freqs_lang(data, lang_vocab):
    freqs = {lang:defaultdict(int) for lang in lang_vocab}
    
    for ex in data:
        lemma, inflection, _, lang, _ = ex
        for ch in list(lemma+inflection):
            freqs[lang][ch] += 1  
    return freqs

    
def main():
    #data_path = 'data_pkl/data_test_release-20khall-special_symbs.pkl'
    data_path = 'data_pkl/data_test_release-10khall-special_symbs.pkl'
    #data_path = 'data_pkl/data_test_release-10khall-all.pkl'
    print('dataset:', data_path)
        
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
    
    proposed_path = f'runs/inflection_langid-opt={args.opt}-lr{args.lr}-wd{args.wd}-bs{args.bs}-d{args.hidden_dim}-orth={args.model_orthogonal_init}-cs={args.initial_curriculum}-cos_att={args.char_att}-{args.curriculum_learning}_{args.label_smoothing}_{args.multitask_learning}_{args.scheduled_sampling}_{args.model_orthogonal_init}'
    while os.path.isdir(proposed_path):
        proposed_path += 'x'
    writer = SummaryWriter(proposed_path)
    
    optimizer = get_opt(model)
    
    #if args.use_curriculum:
    print('Constructing curriculum...')
    lang_char_freqs = compute_freqs_lang(train_data, lang_vocab)
    curriculum = cl.CLLoaderBatchSampling(train_data, char_vocab, tag_vocab, lang_vocab)
    
    if not args.curriculum_learning:
        curriculum.initial_competence = 1.0
        curriculum.competence = 1.0
    else:
        curriculum.get_initial_scores(lang_char_freqs)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps, eta_min=args.min_lr)
    dev_iter = list(lid_data_batcher(dev_data, args.bs, char_vocab, tag_vocab, lang_vocab))
        
    print('Starting training...', args.train_steps)
    model, tloss = trainer_f(args.train_steps, 
                             model, 
                             curriculum, 
                             char_vocab, 
                             writer, 
                             optimizer, 
                             lang_chars, 
                             lang_vocab,
                             lf,
                             lr_scheduler,
                             dev_iter)
        
    test_iter = test_data_batcher(test_data, args.bs, char_vocab, tag_vocab, lang_vocab)
    test(0, model, char_vocab, lang_vocab, test_iter)
    
    return 0

def trainer_f(total_steps, model, curriculum, char_vocab, writer, optimizer, lang_chars, lang_vocab, lf, lr_scheduler, dev_iter):
    model.train()
    curriculum_scores = []
    norms = []
    total_loss = []
    best_dev = 0.0
    
    torch.cuda.empty_cache() # just cleanup...
    
    for ex_i in range(total_steps):
        if curriculum.competence >= 1.0:
            # so we dont have to compute .sum() every time-step
            #if curriculum.times_not_seen[10] == 0:
            #    if curriculum.times_not_seen.sum() == 0:
            #        curriculum.reset_times_not_seen()
            example, batch_weights, batch_idxs = curriculum.sample_with_weight(args.bs)
        else:
            example, batch_weights, batch_idxs = curriculum.sample_batch(args.bs)
        
        print(ex_i, np.round(curriculum.competence, 3), end='\r')
        
        optimizer.zero_grad()
        lemma_mask = (example.lemmas != model.c_pad).int()
        tag_mask = (example.tags != model.t_pad).int()
        target_mask = (example.inflections[:,1:] != model.c_pad).int()
        
        decode_pred, opsy_pred, opsx_pred, wn, ac, at = model(example.lemmas,
                                                         lemma_mask,
                                                         example.tags,
                                                         tag_mask,
                                                         example.inflections,
                                                         test=False)

        batch_langs = example.lemmas[:,0]
        
        if args.label_smoothing:
            decode_loss, gold = get_lbl_smoothing_loss(batch_langs, example, target_mask, decode_pred, lang_chars, char_vocab, batch_weights)
        else:
            decode_loss = _loss(decode_pred, example.inflections[:,1:], F.cross_entropy, model.c_pad)
            gold = example.inflections[:,1:]

        if args.multitask_learning:
            opsy_loss = _loss(opsy_pred, example.ops_y[:,1:], F.cross_entropy, model.o_pad)
            opsx_loss = _loss(opsx_pred, example.ops_x, F.cross_entropy, model.o_pad)
            loss = decode_loss + opsy_loss + opsx_loss
        else:
            opsy_loss = torch.tensor([0])
            opsx_loss = torch.tensor([0])
            loss = decode_loss
        
        # regularization over chars and tags to enforce a more uniform attention distribution
        loss += F.smooth_l1_loss(ac, torch.ones(ac.size(), device=device))
        loss += F.smooth_l1_loss(at, torch.ones(at.size(), device=device))

        loss.backward()
        total_loss.append(loss.item())
        
        # clip grads after gradient calculation
        clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        batch_ex_loss = loss_per_example(decode_pred, example, gold, model.c_pad)
        curriculum_scores += batch_ex_loss
        norms += wn
        
        str_accs, ex_accs = ex_str_accs(decode_pred, example, gold)

        if args.curriculum_learning:
            curriculum.update_competence_sqrt(args.c)
            if args.curriculum_learning_scoring == 'norm':
                curriculum.update_example_scores_norm(wn, batch_idxs)
            elif args.curriculum_learning_scoring == 'loss':
                curriculum.update_example_scores_loss(batch_ex_loss, batch_idxs)
        
        writer.add_scalar('Ops X train loss', opsy_loss.item(), args.c)
        writer.add_scalar('Ops Y train loss', opsx_loss.item(), args.c)
        writer.add_scalar('Decode train loss', decode_loss.item(), args.c)
        writer.add_scalar('Total train loss', loss.item(), args.c)
        writer.add_scalar('Decode train acc', np.mean(str_accs), args.c)
        writer.add_scalar('Decode ex train acc', np.mean(ex_accs), args.c)
        writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], args.c)

        args.c += 1
        
        if (ex_i+1) % 5000 == 0:
            _, _, totals = validate(ex_i+1, model, char_vocab, dev_iter, writer, lf, lang_vocab)
        
            if totals > best_dev:
                best_dev = totals
                print('Saving model at step', ex_i+1)
                save_model(model)
            model.train()
        
    return model, np.mean(total_loss)

def validate(e, model, char_vocab, data_iter, writer, dev_languages, lang_vocab):
    outputs = {lang:{'string_acc':[],
                     'em_acc':[],
                     'pairs':[]} for lang, sizes in dev_languages.items()}
    
    itos = {v:k for k, v in char_vocab.items()}
    lang_vocab_ = {i:x for x, i in lang_vocab.items()}
    out_file = open(f'examples/examples_dev_{e}_{args.curriculum_learning}_{args.label_smoothing}_{args.multitask_learning}_{args.scheduled_sampling}_{args.model_orthogonal_init}.txt', '+w')    
    
    model.eval()
    d_loss_history = []
    d_acc_history = []
    em_acc_history = []
    
    for i, example in enumerate(data_iter):
        if example == []:
            continue
        
        lemma_mask = (example.lemmas != 0).int()
        tag_mask = (example.tags != 0).int()
        with torch.no_grad():
            decode_pred, _, _, _, _, _ = model(example.lemmas,
                                               lemma_mask,
                                               example.tags,
                                               tag_mask,
                                               example.inflections,
                                               tf=False,
                                               test=False)
            
            decode_loss = _loss(decode_pred, example.inflections[:,1:], F.cross_entropy, model.c_pad)
            _, decode_acc = _ex_acc(decode_pred, example.inflections[:,1:]), _acc(decode_pred, example.inflections[:,1:])
            
            preds = torch.argmax(decode_pred, -1)
            for j in range(example.lemmas.size(0)):
                lang_j = lang_vocab_[example.lemmas[j,0].item()]
                gold_j = example.inflections[j,1:]
                input_j = example.lemmas[j,1:]
                pred_j = preds[j]
                non_pad_idxs = len(torch.nonzero(gold_j).squeeze(-1))
                
                #string_acc = (gold_j == pred_j).int().tolist()
                string_nonpad_acc = (gold_j[:non_pad_idxs] == pred_j[:non_pad_idxs]).int().tolist()
                decoded_output = ''.join([itos[x] for x in pred_j.tolist() if x not in [0,1,2]])
                decoded_input = ''.join([itos[x] for x in input_j.tolist() if x not in [0,1,2]])
                decoded_gold = ''.join([itos[x] for x in gold_j.tolist() if x not in [0,1,2]])
                em_acc = int(decoded_gold == decoded_output)
                em_acc_history.append(em_acc)
                
                outputs[lang_j]['string_acc'] += string_nonpad_acc
                outputs[lang_j]['em_acc'].append(em_acc)
                outputs[lang_j]['pairs'].append((decoded_output, decoded_gold))
                
                out_file.write('\t'.join([lang_j, decoded_input, decoded_output, decoded_gold]) + '\n')
        
            d_loss_history.append(decode_loss.item())
            d_acc_history += decode_acc
    
    if writer != False:
        writer.add_scalar('Dev EM', np.mean(em_acc_history), e)
        writer.add_scalar('Dev acc', np.mean(d_acc_history), e)

    totals_em = []
    totals_lev = []
    for k, v in sorted(outputs.items(), key=lambda x: x[0]):
        #em = np.round(np.mean(v['em_acc']), 3)
        #str_acc = np.round(np.mean(v['string_acc']), 3)
        em, lev_dist = eval_form(v['pairs'])
        totals_em.append(em)
        totals_lev.append(lev_dist)
        #if k in ['ame', 'itl', 'ail', 'kod', 'gup', 'sjo', 'ckt', 'vro', 'bra', 'lud', 'evn', 'mag', 'see', 'syc', 'spa', 'deu']:
        print(f'{k} & {em} & {lev_dist} \\\\')
        #print(k, em)
        
    
    print('DEV', e,
          '|',
          np.round(np.mean(d_loss_history), 4), 
          np.round(np.mean(d_acc_history), 4),
          np.round(np.mean(em_acc_history), 4),
          np.round(np.mean(totals_em), 4),
          np.round(np.mean(totals_lev), 4))
    
    #print()
    #format_outputs(outputs)
    return outputs, np.mean(d_loss_history), np.mean(totals_em)

def test(e, model, char_vocab, lang_vocab, data_iter, gold=False):    
    itos = {v:k for k, v in char_vocab.items()}   
    lang_vocab_ = {i:x for x, i in lang_vocab.items()}
    
    test_out = open('examples/test-dev.txt', '+w')
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
            lang_j = lang_vocab_[example.lemmas[j,0].item()]
            prediction_j = ''.join([itos[x] for x in output[0][1:] if x not in [0, 1, 2]])
            lemma_j = ''.join([itos[x.item()] for x in example.lemmas[j,1:] if x not in [0, 1, 2]])
            if gold:
                gold_j = gold[j]
                test_out.write('\t'.join([lang_j, lemma_j, prediction_j, ';'.join(example.str_tags[j])])+'\n')
            else:
                test_out.write('\t'.join([lang_j, lemma_j, prediction_j, ';'.join(example.str_tags[j])])+'\n')
    
    test_out.close()
    return

def _kl_div_loss(pred, gold):
    # mean
    return F.kl_div(pred, gold, reduction='batchmean')

def _kl_div_ex_loss(pred, gold, weights):
    # sum of example loss
    return (F.kl_div(pred, gold, reduction='none').sum(-1).sum(1)*torch.tensor(weights, device=device)).sum()/pred.size(0)

def _loss(pred, gold, loss_fn, ignore_idx=False):
    if ignore_idx:
        return loss_fn(pred.view(-1,pred.size(-1)), gold.contiguous().view(-1), ignore_index=ignore_idx)
    else:
        return loss_fn(pred.view(-1,pred.size(-1)), gold.contiguous().view(-1))

def _acc(pred, gold):
    return (torch.argmax(pred.view(-1,pred.size(-1)),1) == gold.contiguous().view(-1)).int().tolist()

def _ex_acc(pred, gold):
    return (torch.argmax(pred, -1) == gold).int().tolist()

def get_lbl_smoothing_loss(batch_langs, example, target_mask, decode_pred, lang_chars, char_vocab, weights=False):
    with torch.no_grad():
        filter_vecs = lang_chars[batch_langs].squeeze(1)
        smoothed_gold_labels = languagewise_label_smoothing(example.inflections[:,1:], 
                                                            filter_vecs, 
                                                            char_vocab, 
                                                            target_mask, 
                                                            args.smoothing_value)
        
    
    if weights:
        decode_pred = F.log_softmax(decode_pred, -1)
        decode_loss = _kl_div_ex_loss(decode_pred, smoothed_gold_labels, weights)
    else:
        decode_loss = _kl_div_loss(F.log_softmax(decode_pred, -1), smoothed_gold_labels)
    
    gold = torch.argmax(smoothed_gold_labels, 1)
    return decode_loss, gold

def loss_per_example2(decode_pred, example, gold, c_pad):
    return [_loss(decode_pred[m,:], example.inflections[m,1:], F.cross_entropy, c_pad).item() for m in range(gold.size(0))]

def loss_per_example(decode_pred, example, gold, c_pad):
    return [_loss(decode_pred[m,:], example.inflections[m,1:], F.cross_entropy, c_pad).item() for m in range(gold.size(0))]

def ex_str_accs(decode_pred, example, gold):
    non_zero_idx = [len(torch.nonzero(x)) for j, x in enumerate(gold)]
    ex_accs = _ex_acc(decode_pred, example.inflections[:,1:])
    ex_accs = [int(sum(x[:j])==j) for x, j in zip(ex_accs, non_zero_idx)]
    str_accs = _acc(decode_pred, example.inflections[:,1:])
    
    return str_accs, ex_accs

def just_validate():
    pass
    
def ablation_study():
    ablation_dicts = []
    for i, (c, t) in enumerate([(0, False),(0, False),(0, False),(0, False)]):
        args.c = c
        args.curriculum_learning = True
        args.label_smoothing = True
        args.multitask_learning = True
        args.scheduled_sampling = True
        args.model_orthogonal_init = True
        lang_scores = defaultdict(list)
        
        if i == 0:
            print('args.curriculum_learning is false')
            args.curriculum_learning = t
        elif i == 1:
            print('args.label_smoothing is false')
            args.label_smoothing = t
        elif i == 2:
            print('args.multitask_learning is false')
            args.multitask_learning = t
        elif i == 3:
            print('args.scheduled_sampling is false')
            args.scheduled_sampling = t
    
    args.c = 0
    args.curriculum_learning = True
    args.label_smoothing = True
    args.multitask_learning = True
    args.scheduled_sampling = True
    lang_scores = defaultdict(list)
    
    print('NONE is false')
    #for i in range(3):
    lang_scores = main()
    
if __name__ == '__main__':

    #main()
    
    load_and_test_model()
        
    #test_hp()