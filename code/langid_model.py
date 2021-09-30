import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from args import args
from queue import PriorityQueue
import operator
import torch.linalg as linalg

from attention_methods import *

device = torch.device('cuda:0')
EPS = 1e-14
    
class Inflector(nn.Module):
    def __init__(self, start_idx, end_idx, pad_idx, vocab, tags, langs, ops):
        super(Inflector, self).__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.vocab_size = vocab
        self.c_pad = pad_idx[0]
        self.t_pad = pad_idx[1]
        self.o_pad = pad_idx[2]
        self.lang2emb = nn.Embedding(langs, args.embedding_dim) # no padding in languages
        self.char2emb = nn.Embedding(vocab, args.embedding_dim, padding_idx=pad_idx[0])
        self.tag2emb = nn.Embedding(tags, args.embedding_dim, padding_idx=pad_idx[1])
        
        ### scheduled sampling parameters
        self.sc_e = 1.0 # starting probability of grabbing gold
        # if args.sp_step_decay: decay probability for each batch, else decay at new epoch
        # probability decay of grabbing gold, 30,000-ish steps until probability of grabbing gold = 0
        self.sc_decay_rate = self.sc_e/args.prob_decay_steps 
        
        if args.multitask_learning:
            self.ops_discriminator = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                                   nn.Dropout(0.2),
                                                   nn.ReLU(), # ReLU
                                                   nn.Linear(args.hidden_dim, ops, bias=True))
            
        self.char_classifier = nn.Sequential(nn.Linear(args.hidden_dim*4, args.hidden_dim*4),
                                             nn.Dropout(0.3),
                                             nn.ReLU(), # ReLU 
                                             nn.Linear(args.hidden_dim*4, vocab, bias=True))
        
        # dropouts
        self.encoder_drp = nn.Dropout(0.4) # 0.3
        self.decoder_drp = nn.Dropout(0.4) # 0.3
        self.emb_dropout = nn.Dropout(0.2) # 0.1
        
        self.lemma_encoder = nn.LSTM(args.embedding_dim, 
                                     args.hidden_dim, 
                                     batch_first=True, 
                                     bidirectional=True)
        self.tag_encoder = SelfAttentionHead(args.embedding_dim, args.hidden_dim)
        
        self.inflection_decoder = nn.LSTMCell(args.embedding_dim, args.hidden_dim)
        #self.inflection_decoder = nn.LSTMCell(args.hidden_dim*4, args.hidden_dim)
        self.scale_hidden = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        #self.scale_hcx = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        
        if args.char_att == 'cos':
            self.char_attention = CosSimAttention(args.hidden_dim, args.hidden_dim)
        elif args.char_att == 'bah':
            self.char_attention = BahdanauAttention(args.hidden_dim, args.hidden_dim)
        elif args.char_att == 'cos-bah':
            self.char_attention = CosBahAttention(args.hidden_dim, args.hidden_dim)
        else:
            print('Invalid character attention module')
            assert False

        self.tag_attention = BahdanauAttention(args.hidden_dim, args.hidden_dim)
        
    def get_norms(self, output):
        embs = torch.argmax(output, -1)
        embs_norm = linalg.norm(self.char2emb(embs), dim=-1).sum(1)
        return embs_norm
        
    def get_zeros_init(self, bs, dim):
        return torch.zeros((bs, dim), device=device), torch.zeros((bs, dim), device=device)
        
    def forward(self, lemma, lemma_mask, tags, tags_mask, gold=None, tf=True, test=False):
        
        if test:
            return self.inference(lemma, lemma_mask, tags, tags_mask)
        
        # encode input
        lemma_h, tags_h, ht, ct = self.encode_inputs(lemma, lemma_mask, tags, tags_mask)        
        lemma_h = self.scale_hidden(lemma_h)
        
        # predict operations on lemma
        if args.multitask_learning:
            # TODO: attend to tags, ops_lemma = att(lemma_h, tags_h)
            opsx_pred = self.ops_discriminator(lemma_h)
        else:
            opsx_pred = 0
        
        # get input to decoder
        start_t = self.char2emb(lemma[:,0])
        ht, ct = self.get_zeros_init(lemma_h.size(0), args.hidden_dim)

        # decode sequence
        decode_hiddens, decode_pred, atts_c, atts_t = self.batch_decode(start_t, 
                                                                        ht, ct, 
                                                                        lemma_h, lemma_mask, 
                                                                        tags_h, tags_mask, 
                                                                        gold[:,1:],
                                                                        tf)
        if args.multitask_learning:
            decode_hiddens = self.scale_hidden(decode_hiddens)
            opsy_pred = self.ops_discriminator(decode_hiddens)
        else:
            opsy_pred = 0
            
        word_norms = self.get_norms(decode_pred)

        return decode_pred, opsy_pred, opsx_pred, word_norms.tolist(), atts_c, atts_t
    
    def encode_inputs(self, lemma, lemma_mask, tags, tags_mask):
        """
        Encodes lemma and tags, outputs the representations 
        and final hidden states (for the lemma)
        """
        lemma_e = self.encoder_drp(self.char2emb(lemma[:,1:]))
        lang_e = self.lang2emb(lemma[:,0])
        tags_e = self.encoder_drp(self.tag2emb(tags))

        lemma_e = torch.cat([lang_e.unsqueeze(1), lemma_e], 1)
        lemma_h, (ht, ct) = self.lemma_encoder(lemma_e)
        tags_h = self.tag_encoder(tags_e, tags_mask)
        
        return lemma_h, tags_h, ht, ct
        
    def batch_decode(self, x, hx, cx, lemma, lemma_mask, tags, tags_mask, gold, tf=True):
        hidden_states = torch.zeros((gold.size(0), gold.size(1), args.hidden_dim*2), device=device)
        preds = torch.zeros((gold.size(0), gold.size(1), self.vocab_size), device=device)
        att_weights_c = torch.zeros((gold.size(0), gold.size(1), lemma.size(1)), device=device)
        att_weights_t = torch.zeros((gold.size(0), gold.size(1), tags.size(1)), device=device)
        
        for i in range(gold.size(1)):
            hx, cx, p_output, atts_c, atts_t = self.decode_step(x, hx, cx, lemma, lemma_mask, tags, tags_mask)
            hidden_states[:,i,:] = torch.cat([hx, cx], -1)
            preds[:,i,:] = p_output 
            att_weights_c[:,i,:] = atts_c
            att_weights_t[:,i,:] = atts_t
            
            # scheduled sampling (bengio, 2015)
            if tf and self.sc_e > 0.0:
                x = self.scheduled_sampling_(p_output, gold[:,i]) #hcx
                if args.scheduled_sampling:
                    if args.sp_step_decay:
                        self.sc_e = self.sc_e - self.sc_decay_rate if self.sc_e >= 0 else 0
            else:
                y_hat = torch.argmax(F.log_softmax(p_output, 1), 1)
                x = self.emb_dropout(self.char2emb(y_hat))
                
        return hidden_states, preds, att_weights_c, att_weights_t
    
    def decode_step(self, x, hx, cx, lemma, lemma_mask, tags, tags_mask):
        # dropout on input embedding
        x_input = self.decoder_drp(x)
        
        # rnn step
        hx, cx = self.inflection_decoder(x_input, (hx, cx))
        
        # atttention on lemmas and tags
        x_seq_attn, l_atts = self.char_attention(hx, lemma, lemma_mask)
        x_tag_attn, t_atts = self.tag_attention(hx, tags, tags_mask)
        x_output = torch.cat([hx, cx, x_seq_attn, x_tag_attn], -1)   # <--- test     
        p_output = self.char_classifier(x_output)
        
        return hx, cx, p_output, l_atts, t_atts
    
    def scheduled_sampling_(self, p_output, gold):
        probs = torch.rand((p_output.size(0),1), device=device)
        gold_i = self.emb_dropout(self.char2emb(gold))
        p_output_i = self.emb_dropout(self.char2emb(torch.argmax(p_output, 1)))
        x = torch.where(probs > self.sc_e, 
                        p_output_i,
                        gold_i)
        return x
    
    def orthogonal_weight_init_(self, module):
        for weight in module.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
                
    def inference(self, lemma, lemma_mask, tags, tags_mask):
        """
        Inference without teacher forcing and using beam search (DONE?)
        """
        decoded_batch = self.beam_decode(lemma, lemma_mask, tags, tags_mask)
        return decoded_batch
    
    def beam_decode(self, lemma, lemma_mask, tags, tags_mask):
        '''
        adapted version of: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        '''

        beam_width = 2 # 10, 5
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        
        lemma_outputs, tags_outputs, *_ = self.encode_inputs(lemma, lemma_mask, tags, tags_mask)
        start_hx, start_cx = self.get_zeros_init(lemma_outputs.size(0), args.hidden_dim)
        lemma_outputs = self.scale_hidden(lemma_outputs)

        # decoding goes sentence by sentence
        for idx in range(lemma.size(0)):
            sent_lemma_outputs = lemma_outputs[idx,:,:].unsqueeze(0)
            sent_lemma_mask = lemma_mask[idx,:]
            sent_tags_outputs = tags_outputs[idx,:,:].unsqueeze(0)
            sent_tags_mask = tags_mask[idx,:]
            
            hx = start_hx[idx,:].unsqueeze(0)
            cx = start_cx[idx,:].unsqueeze(0)
            
            # Start with the start of the sentence token
            # set fist decoder input to <lang_id>
            decoder_ids = lemma[idx,0]
            decoder_input = self.lang2emb(decoder_ids).unsqueeze(0)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_input, None, decoder_ids, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            k_score, k_node = -node.eval(), node
            nodes.put((k_score, k_node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 1000: break # 2000

                # fetch the best node
                score, n = nodes.get()
                
                # get decoder input from last output
                decoder_input = self.char2emb(n.wordid)
                if len(decoder_input.size()) == 3:
                    decoder_input = decoder_input.squeeze(0)
                elif len(decoder_input.size()) == 1:
                    decoder_input = decoder_input.unsqueeze(0)
                decoder_hidden = n.h

                if n.wordid.item() == self.end_idx and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode one step
                hx, cx, decoder_preds, _, _ = self.decode_step(decoder_input, 
                                                                hx, cx, 
                                                                sent_lemma_outputs, 
                                                                sent_lemma_mask, 
                                                                sent_tags_outputs, 
                                                                sent_tags_mask)
                
                decoder_preds = F.log_softmax(decoder_preds, 1)
                #logp[:,:1] = float('-inf')
                
                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_preds, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1).squeeze(0)
                    log_p = log_prob[0][new_k].item()
                    #print('decoded_t', decoded_t, 'decoder_hidden', decoder_hidden.size())

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                try:
                    endnodes = [nodes.get() for _ in range(topk)]
                except:
                    print('checkp 2')
                    embed()
                    assert False

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch, [0]
           
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    def __lt__(self, other):
        if self.logp < other.logp:
            return True
        else:
            return False
        
    def __le__(self, other):
        if self.logp <= other.logp:
            return True 
        else:
            return False
        
