import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import random
from args import args
from queue import PriorityQueue
import operator
from itertools import takewhile

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
        
        self.ops_discriminator = nn.Linear(args.hidden_dim*2, ops, bias=False)
        self.char_classifier = nn.Sequential(nn.Linear(args.hidden_dim*2, args.hidden_dim*2),
                                             nn.ReLU(), # ReLU
                                             nn.Linear(args.hidden_dim*2, vocab, bias=False))
        
        # dropouts
        self.encoder_drp = nn.Dropout(0.4) # 0.3 # non-sampling dropouts
        self.decoder_drp = nn.Dropout(0.4) # 0.3 # non-sampling dropouts
        self.emb_dropout = nn.Dropout(0.3) # 0.1 # non-sampling dropouts
        
        self.lemma_encoder = nn.LSTM(args.embedding_dim, 
                                     args.hidden_dim, 
                                     batch_first=True, 
                                     bidirectional=True)
        self.tag_encoder = SelfAttentionHead(args.embedding_dim, args.hidden_dim)
        
        self.inflection_decoder = nn.LSTMCell(args.embedding_dim+args.hidden_dim*2, args.hidden_dim)
        self.scale_encoder_outputs = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        
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
        
        
    def get_zeros_init(self, bs, dim):
        return torch.zeros((bs, dim), device=device), torch.zeros((bs, dim), device=device)
        
    def forward(self, lemma, lemma_mask, tags, tags_mask, gold=None, tf=True, test=False):
        
        if test:
            return self.inference(lemma, lemma_mask, tags, tags_mask)
        
        # encode input
        lemma_h, tags_h, *_ = self.encode_inputs(lemma, lemma_mask, tags, tags_mask)
        
        ht, ct = self.get_zeros_init(lemma_h.size(0), args.hidden_dim)
        
        # set first decoder input to <lang>
        start_t = self.char2emb(lemma[:,0])
        
        # predict operations on lemma
        opsx_pred = self.ops_discriminator(lemma_h)
        
        lemma_h = self.scale_encoder_outputs(lemma_h)
        
        # decode sequence
        decode_hiddens, decode_pred = self.batch_decode(start_t, 
                                                        ht, ct, 
                                                        lemma_h, lemma_mask, 
                                                        tags_h, tags_mask, 
                                                        gold[:,1:],
                                                        tf)
             
        opsy_pred = self.ops_discriminator(decode_hiddens)

        return decode_pred, opsy_pred, opsx_pred
    
    def encode_inputs(self, lemma, lemma_mask, tags, tags_mask):
        """
        Encodes lemma and tags, outputs the representations 
        and final hidden states (for the lemma)
        """
        lemma_e = self.encoder_drp(self.char2emb(lemma[:,1:]))
        lang_e = self.encoder_drp(self.char2emb(lemma[:,0]))
        tags_e = self.encoder_drp(self.tag2emb(tags))
        
        lemma_e = torch.cat([lang_e.unsqueeze(1), lemma_e], 1)
        
        lemma_h, (ht, ct) = self.lemma_encoder(lemma_e)
        tags_h = self.tag_encoder(tags_e, tags_mask)
        
        return lemma_h, tags_h, ht, ct
        
    def language_identification(self, hct):
        l_hat = self.lang_discriminator(hct)
        return l_hat
        
    def batch_decode(self, x, hx, cx, lemma, lemma_mask, tags, tags_mask, gold, tf=True):
        hidden_states = torch.zeros((gold.size(0), gold.size(1), args.hidden_dim*2), device=device)
        preds = torch.zeros((gold.size(0), gold.size(1), self.vocab_size), device=device)
        
        for i in range(gold.size(1)):
            hcx, hx, cx, p_hcx = self.decode_step(x, hx, cx, lemma, lemma_mask, tags, tags_mask)
            hidden_states[:,i,:] = hcx
            preds[:,i,:] = p_hcx 
            
            # scheduled sampling (bengio, 2015)
            if tf and self.sc_e > 0.0:
                x = self.scheduled_sampling_(p_hcx, gold[:,i]) #hcx
                if args.sp_step_decay:
                    self.sc_e = self.sc_e - self.sc_decay_rate if self.sc_e >= 0 else 0
            else:
                y_hat = torch.argmax(F.log_softmax(p_hcx, 1), 1)
                x = self.emb_dropout(self.char2emb(y_hat))
                
        return hidden_states, preds
    
    def decode_step(self, x, hx, cx, lemma, lemma_mask, tags, tags_mask):
        #print('decoder hx size:', hx.size())
        # attention from hidden on encoded lemma
        x_seq_attn = self.char_attention(hx, lemma, lemma_mask)
        # attention from hidden_state on encoded tags
        x_tag_attn = self.tag_attention(hx, tags, tags_mask)

        x = torch.cat([x, x_seq_attn, x_tag_attn], -1)        
        x = self.decoder_drp(x)
        
        hx, cx = self.inflection_decoder(x, (hx, cx))
        hcx = torch.cat([hx, cx], -1)
        
        return hcx, hx, cx, self.char_classifier(hcx)
    
    def scheduled_sampling_(self, hcx, gold):
        probs = torch.rand((hcx.size(0),1), device=device)
        gold_i = self.emb_dropout(self.char2emb(gold))
        hcx_i = self.emb_dropout(self.char2emb(torch.argmax(hcx, 1)))
        x = torch.where(probs > self.sc_e, 
                        hcx_i,
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
    
    def no_beam_inference(self, lemma, lemma_mask, tags, tags_mask):
        pass
    
    def beam_decode(self, lemma, lemma_mask, tags, tags_mask):
        '''
        adapted version of: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        '''

        beam_width = 10 # 10
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        
        lemma_outputs, tags_outputs, *_ = self.encode_inputs(lemma, lemma_mask, tags, tags_mask)
        start_hx, start_cx = self.get_zeros_init(lemma_outputs.size(0), args.hidden_dim)
        lemma_outputs = self.scale_encoder_outputs(lemma_outputs)

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
                if qsize > 2000: break

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
                decoder_hidden, hx, cx, decoder_preds = self.decode_step(decoder_input, 
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
        
class CosBahAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CosBahAttention, self).__init__()
        self.cos = CosSimAttention(in_dim, out_dim)
        self.bah = BahdanauAttention(in_dim, out_dim)
        self.linear_combine = nn.Linear(out_dim*2, out_dim)
    
    def forward(self, k, xs, mask):
        cos_repr = self.cos(k, xs, mask)
        bah_repr = self.bah(k, xs, mask)
        attn = self.linear_combine(torch.cat([cos_repr, bah_repr], -1))
        return attn

class CosSimAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        compute cosine similarity between previous hidden_state and encoded lemma sequence
        train the model such that the hidden-state generated by the decoder-lstm matches 
          the lemma in the cases of copy
        ~= a pointer network
        """
        super(CosSimAttention, self).__init__()
        self.softabs = lambda x, epsilon: torch.sqrt(torch.pow(x, 2.0) + epsilon)
        self.eps = 1e-3 # 1e-14
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, k, xs, mask):
        cos_sim = self.cos(k.unsqueeze(1).repeat(1,xs.size(1),1), xs)
        a_cos = self.softabs(cos_sim, self.eps) * mask + EPS
        attn = torch.einsum('bs,bsk->bk', [a_cos, xs])
        return attn
    
class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention, self).__init__()
        self.lin = nn.Linear(out_dim, out_dim)
        self.eps = 1e-14
        
    def forward(self, k, xs, mask):
        #xs = self.scale(xs)
        a = F.softmax(torch.einsum('bw,bsk->bs', [k, xs]),-1) * mask + EPS
        attn = torch.einsum('bi,bik->bk', [a, xs])
        return attn
    
class BahdanauAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BahdanauAttention, self).__init__()
        self.energy = nn.Parameter(torch.randn(in_dim))
        self.Wa = nn.Linear(in_dim, in_dim)
        self.Wb = nn.Linear(in_dim, in_dim)
        
    def forward(self, k, xs, mask):
        ks = k.unsqueeze(1).repeat(1,xs.size(1),1)
        w = torch.tanh(self.Wa(ks) + self.Wb(xs))
        # calculate "energy"
        attn = F.softmax(w @ self.energy, -1) * mask + EPS
        return torch.einsum('bi,bik->bk', [attn, xs])
    
class SelfAttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttentionHead, self).__init__()
        self.k = nn.Linear(in_dim, out_dim) # remove bias here
        self.q = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.lin = nn.Linear(out_dim, out_dim)
        
    def forward(self, xs, mask):
        qk = torch.einsum('bsd,bde->bse', [self.k(xs), self.q(xs).transpose(2,1)])
        mask = mask.repeat(1,qk.size(1)).view(qk.size())
        a = F.softmax(qk/xs.size(-1), 1) * mask + EPS
        attn = torch.einsum('bij,bik->bik', [a, self.v(xs)])
        attn = F.leaky_relu(self.lin(attn))
        return attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, in_dim, out_dim):
        super(MultiHeadAttention, self).__init__()
        self.out_dim = out_dim
        self.dropout = nn.Dropout(.33)
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(in_dim, out_dim) for _ in range(num_heads)])
        
        self.ffn = nn.Sequential(nn.Linear(out_dim*4, out_dim*4, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(out_dim*4, out_dim, bias=False))
        
    def forward(self, xs, mask):
        container = torch.zeros((self.num_heads, xs.size(0), xs.size(1), self.out_dim), device=device)
        for i, head in enumerate(self.heads):
            container[i,:,:,:] = head(xs, mask)
            
        container = torch.cat([container[i,:] for i in range(self.num_heads)], -1)
        
        return self.ffn(container)
            
class MultiLayerEncoder(nn.Module):
    def __init__(self, num_heads, num_layers, in_dim, out_dim):
        super(MultiLayerEncoder, self).__init__()
        
        setups = [(in_dim, out_dim)] + [(out_dim, out_dim)]*num_layers
        self.layers = nn.ModuleList([MultiHeadAttention(num_heads, i_d, o_d) for i_d, o_d in setups])
        self.dropout = nn.Dropout(.33)
        self.final_ffn = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False),
                                       nn.Tanh(),
                                       nn.Linear(out_dim, out_dim, bias=False))
    
    def forward(self, xs, mask):
        for layer in self.layers:
            xs = self.dropout(xs)
            xs = layer(xs, mask)
        
        xs = self.dropout(xs)
        return self.final_ffn(xs)
    
class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super(SelfAttentionEncoder, self).__init__()
        self.dim = dim
        self.k = nn.Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.lin = nn.Linear(dim, dim)
        
    def forward(self, xs):
        qk = torch.einsum('bsd,bde->bse', [self.k(xs), self.q(xs).transpose(2,1)])
        a = F.softmax(qk/self.dim, 1)
        attn = torch.einsum('bij,bik->bik', [a, self.v(xs)])
        return torch.tanh(self.lin(attn))
    
class DropoutLSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttentionEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def forward(self, x):
        pass