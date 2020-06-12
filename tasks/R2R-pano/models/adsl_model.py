import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from models.modules import build_mlp, SoftAttention, PositionalEncoding, ScaledDotProductAttention, create_mask, proj_masking, PositionalEncoding
from models.rnn import TranslatorEncoderRNN, DecoderRNN, SpeakerEncoderRNN, SpeakerDecoderLSTM
from models.transformer import DecoderTransformer
from models.tcn import TranslatorTCN, TranslatorTCNSkip
from utils import Entropy, calculate_pis, end_token_idx, padding_idx, period_idx

import pdb

class Translator(nn.Module):
    """ Neural Machine Translation that uses Seq2seq model to translate src language into tgt language. """

    def __init__(self, encoder_kwargs, decoder_kwargs, gumbel_hard=False, neuralese_len=None,
                 decoder_option='transformer', token_entropy_beta=0, is_backtranslator=False, 
                 segment_by_sent=False, segment_ctx=False, num_neuralese=4, max_num_sentences=6,
                 greedy_decode=False):
        super(Translator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neuralese_len = neuralese_len 
        self.encoder = TranslatorEncoderRNN(**encoder_kwargs)
        self.lang_position = PositionalEncoding(encoder_kwargs['hidden_size'], dropout=0.1, max_len=80)
        self.decoder_option = decoder_option
        self.token_entropy_beta = token_entropy_beta
        self.is_backtranslator = is_backtranslator
        self.segment_by_sent = segment_by_sent
        self.segment_ctx = segment_ctx 
        self.encoder = TranslatorEncoderRNN(**encoder_kwargs)
        self.share_embedding = decoder_kwargs['share_embedding']
        self.use_end_token      = decoder_kwargs['use_end_token']
        self.num_neuralese=num_neuralese
        self.max_num_sentences=max_num_sentences
        self.decode_greedy = greedy_decode
        if self.decode_greedy:
            print('Use greedy decoding in Translator ******')
        if self.use_end_token:
            assert(gumbel_hard == True)
        if decoder_option == 'rnn':
            self.decoder = DecoderRNN(**decoder_kwargs)
        elif decoder_option == 'transformer' or decoder_option == 'transformer-parallel':
            self.decoder = DecoderTransformer(**decoder_kwargs)
            self.word_layer = nn.Linear(decoder_kwargs['vocab_size'], decoder_kwargs['hidden_size'], bias=False)
            self.word_init  = nn.Linear(decoder_kwargs['hidden_size'], decoder_kwargs['vocab_size'])
            self.ctx_layer  = nn.Linear(decoder_kwargs['hidden_size'], decoder_kwargs['hidden_size'])
            self.position_layer = nn.Linear(1, decoder_kwargs['hidden_size'] // 2)
            self.h_layer = nn.Linear(decoder_kwargs['hidden_size'], decoder_kwargs['hidden_size'] // 2)
        elif decoder_option == 'tcn':
            self.decoder = TranslatorTCN(**decoder_kwargs)
        elif decoder_option == 'tcn_skip':
            self.decoder = TranslatorTCNSkip(**decoder_kwargs)
        print('Neuralese length: ', neuralese_len)
        print('Neuralese vocab size: ', self.decoder.vocab_size)
        print('Translator entropy beta: ', token_entropy_beta)

        self.entropy     = Entropy()
        self.tau0        = 1.0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP    = 0.5
        self.hard        = gumbel_hard

    def update_tau(self, iteration):
        self.tau = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * iteration), self.MIN_TEMP)

    def _decode_rnn(self, h_t, c_t, ctx, ctx_mask, provided_pred_seq=None, neuralese_len_customized=None, end_word_prior=0, train=True):
        # rollout translation
        pred_seq = []
        logits = []
        pred_seq_lengths = torch.zeros(ctx.shape[0]).to(self.device)
        w_t = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)

        entropy_total = 0
        end_word = torch.zeros(ctx.shape[0], 1).to(self.device)
        end_word = ((end_word + end_word_prior) > 0).float()
        neuralese_len = self.neuralese_len
        if neuralese_len_customized != None:
          neuralese_len = neuralese_len_customized
        for i in range(0, neuralese_len):
          if self.use_end_token:
            end_word = ((end_word + w_t.detach()[:,end_token_idx].unsqueeze(-1) == 1) > 0).float()
          h_t, c_t, alpha, logit_t   = self.decoder(w_t, h_t, c_t, ctx, ctx_mask)
          logit_t = logit_t * (1 - end_word).expand(logit_t.shape)

          # If we are given a pred_seq to calculate probability for, use that.
          if provided_pred_seq is not None:
            w_t = provided_pred_seq[:,i,:]
          else: 
            if self.hard:
              if train or not self.decode_greedy:
                w_t     = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)
              else:
                max_results = logit_t.max(dim=1)
                w_t     = torch.eye(self.decoder.vocab_size)[max_results[1]].to(self.device).detach()
            else:
              w_t = F.softmax(logit_t, dim=1)
          w_t = w_t * (1 - end_word).expand(w_t.shape)

          entropy_total += self.entropy(logit_t)
          pred_seq += [w_t.unsqueeze(1)]
          logits   += [logit_t.unsqueeze(-1)]
          pred_seq_lengths = pred_seq_lengths + (1 - end_word[:,0])
          # Calculate probability

        pred_seq = torch.cat(pred_seq, dim=1)  
        logits   = torch.cat(logits, dim=-1)
        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, logits 


    def _get_sentence_masks(self, reset_indices):
        # This is about to get super ugly.
        # We need to construct the sentence mask based on reset_indices, but if we do in place matrix
        # assignments we're probably going to get an error from torch. So I'm going to do everything 
        # in numpy and then initialize a new torch tensor.
        # First, create a numpy matrix of batch_size x max_num sentence. mat[i,j+1] corresponds to where
        # the j^th sentence of example i ends. First index = 0 for all instructions to mark starting loc.
        # tasks/R2R-pano/dummy, python mask.py gives an example of how this works
        reset_indices = reset_indices.cpu().numpy()
        sent_end_mat = np.zeros((reset_indices.shape[0], self.max_num_sentences+1), dtype=np.uint8)
        j_counter = np.ones(reset_indices.shape[0], dtype=np.uint8) # to count which sentence we're on for instruction i
        rows, cols = reset_indices.nonzero() # gets locations of where sentences end. 
        for r, c in zip(rows, cols):
            try:
              sent_end_mat[r, j_counter[r]] = c+1 # set mat[i,j] to ind where sentence ends
            except:
              import pdb; pdb.set_trace()
            j_counter[r] += 1 # keep track of j
        sent_end_mat[sent_end_mat == 0] = reset_indices.shape[1] # Any un-used slots are set to the end of ctx matrix.
        sent_end_mat[:,0] = 0 # except for the first column
        # Now, we can loop through each sentence ind, create the appropriate sentence mask, multiply
        # it with ctx_mask to get the values we should be looking at, and generate Neuralese for the sentence.
        sentence_masks = []
        for sent_ind in range(self.max_num_sentences):
            sentence_mask = np.zeros(reset_indices.shape)
            for instr_ind in range(reset_indices.shape[0]):
                # Next lne sets sentence masks
                sentence_mask[instr_ind, sent_end_mat[instr_ind,sent_ind]:sent_end_mat[instr_ind,sent_ind+1]] = 1
            sentence_masks.append(sentence_mask)
        return sentence_masks

    def _get_mask_end_indices(self, sentence_mask):
        mask_end_indices = [0] * sentence_mask.shape[0]
        for i in range(sentence_mask.shape[0]):
            for j in range(sentence_mask.shape[1]):
                if sentence_mask[i,j] == 1:
                    mask_end_indices[i] = j
        return mask_end_indices
        
    def _decode_rnn_segment(self, h_t, c_t, ctx_c, ctx, ctx_mask, reset_indices, provided_pred_seq=None):
        pred_seq = []
        logits   = []
        entropy_total = 0
        pred_seq_lengths =  torch.zeros(ctx.shape[0]).to(self.device)
       
        sentence_masks = self._get_sentence_masks(reset_indices)
        for sentence_mask in sentence_masks: 
            sentence_mask = torch.tensor(sentence_mask).byte().to(self.device)
            mask_end_indices = self._get_mask_end_indices(sentence_mask)
            ctx_mask = ctx_mask * sentence_mask
            # generate length 4 neuralese, attach onto pred_seq, etc etc. 
            h_t_segment = ctx[torch.arange(h_t.shape[0]), mask_end_indices]
            c_t_segment = ctx_c[torch.arange(h_t.shape[0]), mask_end_indices]
            outputs     = self._decode_rnn(h_t_segment, c_t_segment, ctx, ctx_mask, \
                                           provided_pred_seq=None, \
                                           neuralese_len_customized=self.num_neuralese, \
                                           end_word_prior=(sentence_mask.float().sum(-1, keepdim=True) == 0).float())
            segment_pred_seq, \
            segment_pred_seq_lengths, \
            segment_entropy_total, \
            segment_logits               = outputs

            pred_seq += [segment_pred_seq]
            pred_seq_lengths += (sentence_mask.float().sum(-1) > 0).float() * torch.Tensor(segment_pred_seq_lengths).cuda()
            entropy_total += segment_entropy_total
            logits += [segment_logits]

        pred_seq = torch.cat(pred_seq, dim=1)
        logits   = torch.cat(logits, dim=-1)

        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, logits
                 

    def _decode_transformer(self, ctx, ctx_mask, h_init=None, provided_pred_seq=None):
        assert(self.use_end_token==False)
        pred_seq = []
        pred_seq_lengths = torch.zeros(ctx.shape[0]).to(self.device)
        if h_init is None:
          w_t = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)
        else:
          w_t = self.word_init(h_init)
        w_t_mask  = torch.ones(ctx.shape[0], 1).byte().to(self.device)

        ctx = self.ctx_layer(ctx)
        ctx = ctx * ctx_mask.float().unsqueeze(-1).expand(ctx.shape)
        entropy_total = 0
        share_embedding = self.share_embedding
        log_prob = torch.zeros(ctx.shape[0]).to(self.device)
        for i in range(0, self.neuralese_len):
          # update ctx and ctx mask
          w_t_embedding = self.word_layer(w_t)
          ctx      = torch.cat([ctx, w_t_embedding.unsqueeze(1)], dim=1)
          ctx_mask = torch.cat([ctx_mask, w_t_mask], dim=1)

          # pass through transformer
          logit_t = self.decoder(ctx, ctx_mask)
          entropy_total += self.entropy(logit_t)

          # sample words from distribution
          if provided_pred_seq is not None:
            w_t = provided_pred_seq[:,i,:]
          else: 
            if self.hard:
              w_t       = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)
            else:
              w_t       = F.softmax(logit_t, dim=1)

          #w_t       = F.softmax(logit_t, dim=-1)
          w_t_mask  = torch.ones(ctx.shape[0], 1).byte().to(self.device)
          # update outputs
          if not share_embedding:
            pred_seq += [w_t.unsqueeze(1)]
          else:
            pred_seq += [self.word_layer(w_t).unsqueeze(1)]
          pred_seq_lengths = pred_seq_lengths + 1
          # Calculate probability
          pis     = calculate_pis(logit_t)
          log_prob     += torch.sum(pis * w_t, dim=-1).log()

        pred_seq = torch.cat(pred_seq, dim=1)  

        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, log_prob

    def _decode_transformer_parallel(self, ctx, ctx_mask, h_init=None, provided_pred_seq=None):
        assert(self.use_end_token==False)
        pred_seq = []
        pred_seq_lengths = self.neuralese_len * torch.ones(ctx.shape[0]).to(self.device)
        word_mask = torch.ones(ctx.shape[0], self.neuralese_len).byte().to(self.device)
        positions = (torch.arange(0, self.neuralese_len).to(self.device)).float()
        positions = positions.unsqueeze(-1).unsqueeze(0).expand(ctx.shape[0], self.neuralese_len, -1)
        positions = self.position_layer(positions)
        h_init    = self.h_layer(h_init).unsqueeze(1).expand(ctx.shape[0], self.neuralese_len, -1)
        word_init = torch.cat([positions, h_init], dim=-1)

        mask = torch.cat([ctx_mask, word_mask], dim=1)

        h = self.decoder(torch.cat([ctx, word_init], dim=1), mask, return_full=True)

        logits = h[:,-self.neuralese_len:,:]

        entropy_total = self.entropy(logits.contiguous().view(-1, logits.shape[-1])) * self.neuralese_len
        if provided_pred_seq is not None:
          pred_seq = provided_pred_seq
        else: 
          pred_seq = F.gumbel_softmax(logits.contiguous().view(-1, logits.shape[-1]), hard=self.hard, tau=self.tau)
          pred_seq = pred_seq.view(logits.shape)
        #pred_seq = logits

        # Calculate probabilities from logits and chosen values
        log_prob = torch.zeros(ctx.shape[0]).to(self.device)
        for t in range(logits.size()[1]):
            logit_t = logits[:,t,:]
            pis     = calculate_pis(logit_t)
            w_t     = pred_seq[:,t,:]
            log_prob   += torch.sum(pis * w_t, dim=-1).log()
                
        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, log_prob

    def _decode_tcn(self, ctx, ctx_mask, seq_lengths):
        # get translation
        logit, pred_seq_lengths = self.decoder(ctx, seq_lengths)
        N, T, C = logit.shape
        logit_2d = logit.view(N * T, C)
        #pred_seq     = F.gumbel_softmax(logit_2d, hard=self.hard, tau=self.tau)
        pred_seq = logit_2d
        pred_seq = pred_seq.view(N, T, C)
        return pred_seq, pred_seq_lengths, 0, 1

    def _decode_tcn_skip(self, ctx, ctx_mask, seq_lengths):
        # get translation
        logit, pred_seq_lengths = self.decoder(ctx, seq_lengths, skip=True)
        #N, T, C = logit.shape
        #logit_2d = logit.view(N * T, C)
        #pred_seq     = F.gumbel_softmax(logit_2d, hard=self.hard, tau=self.tau)
        return logit, pred_seq_lengths, 0, 1

    def forward(self, seq, seq_lengths, src_lang=None, provided_pred_seq=None, train=True):
        if self.segment_by_sent and self.segment_ctx:
            # Marks where periods are.
            reset_indices = (seq == period_idx).float()
            ctx, ctx_c, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths, reset_indices, return_c=True)
        else:
            # Otherwise, we never reset
            reset_indices = None
            ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths, reset_indices=None)
        #ctx = self.lang_position(ctx)
        ctx_mask = (ctx_mask == 1).byte()

        # rollout translation
        if self.decoder_option == 'rnn' and not self.segment_by_sent:
          pred_seq, pred_seq_lengths, entropy_total, logits = self._decode_rnn(h_t, c_t, ctx, ctx_mask, provided_pred_seq=provided_pred_seq, train=train)
        if self.decoder_option == 'rnn' and self.segment_by_sent:
          pred_seq, pred_seq_lengths, entropy_total, logits = self._decode_rnn_segment(h_t, c_t, ctx, ctx_c, ctx_mask, reset_indices, provided_pred_seq=provided_pred_seq)
        elif self.decoder_option == 'transformer':
          pred_seq, pred_seq_lengths, entropy_total, logits = self._decode_transformer(ctx, ctx_mask, h_t, provided_pred_seq=provided_pred_seq)
        elif self.decoder_option == 'transformer-parallel':
          pred_seq, pred_seq_lengths, entropy_total, logits = self._decode_transformer_parallel(ctx, ctx_mask, h_t, provided_pred_seq=provided_pred_seq)
        elif self.decoder_option == 'tcn':
          pred_seq, pred_seq_lengths, entropy_total, prob = self._decode_tcn(ctx, ctx_mask, seq_lengths)
        elif self.decoder_option == 'tcn_skip':
          pred_seq, pred_seq_lengths, entropy_total, prob = self._decode_tcn_skip(ctx, ctx_mask, seq_lengths)
        elif self.decoder_option == 'transformer_backtrans':
          raise ValueError('Not implemented.')

        return pred_seq, pred_seq_lengths, - self.token_entropy_beta * entropy_total, logits


class Speaker(nn.Module):
    """ Speaker that uses Seq2seq model to translate visual trajectories into languages. """
    def __init__(self, encoder_kwargs, decoder_kwargs, neuralese_len, gumbel_hard=False,
                 token_entropy_beta=0, var_length_neuralese=False, compositional=False, 
                 segment=False, segment_ctx=False, segment_prob=0, num_neuralese=4, max_num_sentences=6):
        super(Speaker, self).__init__()
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder,           \
        self.decoder              = self._init_encoder_decoder(encoder_kwargs, decoder_kwargs)
        self.neuralese_len        = neuralese_len
        self.token_entropy_beta   = token_entropy_beta
        self.var_length_neuralese = var_length_neuralese
        self.compositional        = compositional
        self.segment              = segment
        self.segment_ctx          = segment_ctx
        self.segment_prob         = segment_prob
        self.num_neuralese=num_neuralese
        self.max_num_sentences=max_num_sentences
        
        self.entropy     = Entropy()
        self.tau0        = 1.0
        self.ANNEAL_RATE = 0.000003
        self.MIN_TEMP    = 0.5
        self.hard        = gumbel_hard
        
    def _init_encoder_decoder(self, encoder_kwargs, decoder_kwargs):
        encoder = SpeakerEncoderRNN(**encoder_kwargs)
        decoder = DecoderRNN(**decoder_kwargs)
        encoder.to(self.device)
        decoder.to(self.device)
        return encoder, decoder
    
    def update_tau(self, iteration):
        self.tau = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * iteration), self.MIN_TEMP)

    ''' copied from Translator '''        
    def _decode_rnn(self, h_t, c_t, ctx, ctx_mask, provided_pred_seq=None, neuralese_len_customized=None, end_word_prior=0):
        # rollout translation
        pred_seq = []
        logits = []
        pred_seq_lengths = torch.zeros(ctx.shape[0]).to(self.device)
        w_t = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)

        entropy_total = 0
        end_word = torch.zeros(ctx.shape[0], 1).to(self.device)
        end_word = ((end_word + end_word_prior) > 0).float()
        neuralese_len = self.neuralese_len
        if neuralese_len_customized != None:
          neuralese_len = neuralese_len_customized
        for i in range(0, neuralese_len):
          h_t, c_t, alpha, logit_t   = self.decoder(w_t, h_t, c_t, ctx, ctx_mask)
          logit_t = logit_t * (1 - end_word).expand(logit_t.shape)

          # If we are given a pred_seq to calculate probability for, use that.
          if provided_pred_seq is not None:
            w_t = provided_pred_seq[:,i,:]
          else: 
            w_t     = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)
          w_t = w_t * (1 - end_word).expand(w_t.shape)

          entropy_total += self.entropy(logit_t)
          pred_seq += [w_t.unsqueeze(1)]
          logits   += [logit_t.unsqueeze(-1)]
          pred_seq_lengths = pred_seq_lengths + (1 - end_word[:,0])
          # Calculate probability

        pred_seq = torch.cat(pred_seq, dim=1)  
        logits   = torch.cat(logits, dim=-1)
        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, logits 

    ''' copied from Translator '''        
    def _get_sentence_masks(self, reset_indices):
        # This is about to get super ugly.
        # We need to construct the sentence mask based on reset_indices, but if we do in place matrix
        # assignments we're probably going to get an error from torch. So I'm going to do everything 
        # in numpy and then initialize a new torch tensor.
        # First, create a numpy matrix of batch_size x max_num sentence. mat[i,j+1] corresponds to where
        # the j^th sentence of example i ends. First index = 0 for all instructions to mark starting loc.
        # tasks/R2R-pano/dummy, python mask.py gives an example of how this works
        reset_indices = reset_indices.cpu().numpy()
        sent_end_mat = np.zeros((reset_indices.shape[0], self.max_num_sentences+1), dtype=np.uint8)
        j_counter = np.ones(reset_indices.shape[0], dtype=np.uint8) # to count which sentence we're on for instruction i
        rows, cols = reset_indices.nonzero() # gets locations of where sentences end. 
        for r, c in zip(rows, cols):
            sent_end_mat[r, j_counter[r]] = c+1 # set mat[i,j] to ind where sentence ends
            j_counter[r] += 1 # keep track of j
        sent_end_mat[sent_end_mat == 0] = reset_indices.shape[1] # Any un-used slots are set to the end of ctx matrix.
        sent_end_mat[:,0] = 0 # except for the first column
        # Now, we can loop through each sentence ind, create the appropriate sentence mask, multiply
        # it with ctx_mask to get the values we should be looking at, and generate Neuralese for the sentence.
        sentence_masks = []
        for sent_ind in range(self.max_num_sentences):
            sentence_mask = np.zeros(reset_indices.shape)
            for instr_ind in range(reset_indices.shape[0]):
                # Next lne sets sentence masks
                sentence_mask[instr_ind, sent_end_mat[instr_ind,sent_ind]:sent_end_mat[instr_ind,sent_ind+1]] = 1
            sentence_masks.append(sentence_mask)
        return sentence_masks

    ''' copied from Translator '''        
    def _get_mask_end_indices(self, sentence_mask):
        mask_end_indices = [0] * sentence_mask.shape[0]
        for i in range(sentence_mask.shape[0]):
            for j in range(sentence_mask.shape[1]):
                if sentence_mask[i,j] == 1:
                    mask_end_indices[i] = j
        return mask_end_indices

    ''' copied from Translator '''        
    def _decode_rnn_segment(self, h_t, c_t, ctx_c, ctx, ctx_mask, reset_indices, provided_pred_seq=None):
        pred_seq = []
        logits   = []
        entropy_total = 0
        pred_seq_lengths =  torch.zeros(ctx.shape[0]).to(self.device)
        
        sentence_masks = self._get_sentence_masks(reset_indices)
        for sentence_mask in sentence_masks: 
            sentence_mask = torch.tensor(sentence_mask).byte().to(self.device)
            mask_end_indices = self._get_mask_end_indices(sentence_mask)
            ctx_mask = ctx_mask * sentence_mask
            # generate length 4 neuralese, attach onto pred_seq, etc etc. 
            h_t_segment = ctx[torch.arange(h_t.shape[0]), mask_end_indices]
            c_t_segment = ctx_c[torch.arange(h_t.shape[0]), mask_end_indices]
            outputs     = self._decode_rnn(h_t_segment, c_t_segment, ctx, ctx_mask, \
                                           provided_pred_seq=None, \
                                           neuralese_len_customized=self.num_neuralese, \
                                           end_word_prior=(sentence_mask.float().sum(-1, keepdim=True) == 0).float())
            segment_pred_seq, \
            segment_pred_seq_lengths, \
            segment_entropy_total, \
            segment_logits               = outputs

            pred_seq += [segment_pred_seq]
            pred_seq_lengths += (sentence_mask.float().sum(-1) > 0).float() * torch.Tensor(segment_pred_seq_lengths).cuda()
            entropy_total += segment_entropy_total
            logits += [segment_logits]

        pred_seq = torch.cat(pred_seq, dim=1)
        logits   = torch.cat(logits, dim=-1)

        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, logits

    def forward_compositional_by_segment(self, all_img_feats, all_action_embeddings, provided_pred_seq):
        ctx,      \
        ctx_c,    \
        h_t,      \
        c_t,      \
        ctx_mask, \
        reset_indices = self.encoder(all_img_feats, all_action_embeddings, return_c=True, \
                                     segment=[self.segment, self.segment_ctx, self.segment_prob])
        ctx_mask = (ctx_mask == 1).byte()
 
        pred_seq, \
        pred_seq_lengths, \
        entropy_total, \
        logits            = self._decode_rnn_segment(h_t, c_t, ctx_c, ctx, ctx_mask, reset_indices, provided_pred_seq=None)

        return pred_seq, pred_seq_lengths, - self.token_entropy_beta * entropy_total, logits 

    def forward_standard(self, all_img_feats, all_action_embeddings, provided_pred_seq):
        ctx,     \
        h_t,     \
        c_t,     \
        ctx_mask = self.encoder(all_img_feats, all_action_embeddings)
        ctx_mask = (ctx_mask == 1).byte()
        if self.var_length_neuralese:
            path_lengths      = ctx_mask.sum(dim=1).detach() - 1
            neuralese_lengths = path_lengths * 2
        w_t      = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)
        pred_seq = []
        logits   = []
        pred_seq_lengths = torch.zeros(ctx.shape[0]).to(self.device)
        entropy_total = 0
        for i in range(0, self.neuralese_len):
            h_t, c_t, alpha, logit_t = self.decoder(w_t, h_t, c_t, ctx, ctx_mask)
            if self.var_length_neuralese:
                mask = (i < neuralese_lengths).float().unsqueeze(-1).expand(logit_t.shape)
                logit_t = logit_t * mask

            if provided_pred_seq is not None:
                w_t = provided_pred_seq[:,i,:]
            else:
                if self.hard:
                    w_t = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)
                else:
                    w_t = F.softmax(logit_t, dim=1)
                if self.var_length_neuralese:
                    w_t = w_t * mask 
                
            # w_t = logit_t --> seems there's no big difference
            entropy_total           += self.entropy(logit_t)
            pred_seq                += [w_t.unsqueeze(1)]
            logits                  += [logit_t.unsqueeze(-1)]
            pred_seq_lengths         = pred_seq_lengths + 1
             
        pred_seq = torch.cat(pred_seq, dim=1)
        logits   = torch.cat(logits, dim=-1)
        if self.var_length_neuralese:
            pred_seq_lengths = neuralese_lengths
        return pred_seq, pred_seq_lengths.long().tolist(), - self.token_entropy_beta * entropy_total, logits 

    def forward_compositional(self, all_img_feats, all_action_embeddings, provided_pred_seq=None):
        print('Generating Neuralese in a compositional way')
        batch_size       = len(all_img_feats)
        pred_seq         = []
        logits           = []
        pred_seq_lengths = torch.zeros(batch_size).to(self.device)
        entropy_total    = 0

        for i in range(self.neuralese_len // 2):
            ended_np = np.zeros((batch_size, 1))
            one_step_img_feats = []
            one_step_action_embeddings = []
            # Grab image features for that step
            for j in range(batch_size):
                img_feats = all_img_feats[j]
                action_embeds = all_action_embeddings[j]
                if i < len(img_feats) - 1: # If we haven't finished the path, get the next step
                    one_step_img_feats.append([img_feats[i], img_feats[i+1]])
                    one_step_action_embeddings.append([action_embeds[i], action_embeds[i+1]])
                else: # If we already finished the path, get the last step as a placeholder
                    one_step_img_feats.append([img_feats[-2], img_feats[-1]])
                    one_step_action_embeddings.append([action_embeds[-2], action_embeds[-1]])
                    ended_np[j,0] = 1
            ended = torch.from_numpy(ended_np).float().to(self.device)
            # Feed these one step paths and get tokens and logits
            ctx, h_t, c_t, ctx_mask = self.encoder(one_step_img_feats, one_step_action_embeddings)
            ctx_mask = (ctx_mask == 1).byte()
            w_t = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)
            for t in range(0, 2):
                h_t, c_t, alpha, logit_t = self.decoder(w_t, h_t, c_t, ctx, ctx_mask)
                mask = (1 - ended).expand(logit_t.shape)
                logit_t = logit_t * mask
                if provided_pred_seq is not None:
                    w_t = provided_pred_seq[:, i*2+t, :]
                else:
                    if self.hard:
                        w_t = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)
                    else:
                        w_t = F.softmax(logit_t, dim=1)
                    w_t = w_t * mask
                entropy_total += self.entropy(logit_t)
                pred_seq      += [w_t.unsqueeze(1)]
                logits        += [logit_t.unsqueeze(-1)]
                pred_seq_lengths = pred_seq_lengths + (1 - ended[:,0])
            
        pred_seq = torch.cat(pred_seq, dim=1)
        logits   = torch.cat(logits, dim=-1)
        return pred_seq, pred_seq_lengths.long().tolist(), - self.token_entropy_beta * entropy_total, logits 
    
    def forward(self, all_img_feats, all_action_embeddings, provided_pred_seq=None):
        if self.compositional:
            pred_seq, \
            pred_seq_lengths, \
            extra_loss, \
            logits = self.forward_compositional(all_img_feats, all_action_embeddings, provided_pred_seq)
        elif self.segment:
            pred_seq, \
            pred_seq_lengths, \
            extra_loss, \
            logits = self.forward_compositional_by_segment(all_img_feats, all_action_embeddings, provided_pred_seq)
        else:
            pred_seq, \
            pred_seq_lengths, \
            extra_loss, \
            logits = self.forward_standard(all_img_feats, all_action_embeddings, provided_pred_seq)
        return pred_seq, pred_seq_lengths, extra_loss, logits

class NeuraleseGenerator(nn.Module):
    """ Neural Machine Translation that uses Seq2seq model to translate src language into tgt language. """

    def __init__(self, encoder_type, encoder_kwargs, decoder_kwargs, gumbel_hard=False, neuralese_len=None,
                 decoder_option='transformer', token_entropy_beta=0):
        super(NeuraleseGenerator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Encoder based on whether the model is a translator or speaker
        assert encoder_type in ['translator', 'backtranslator', 'speaker']
        self.encoder_type = encoder_type
        if self.encoder_type in ['translator', 'backtranslator']:
            self.encoder = TranslatorEncoderRNN(**encoder_kwargs)
        elif self.encoder_type in ['speaker']:
            self.encoder = SpeakerEncoderRNN(**encoder_kwargs)

        # Initialize Decoder given specifications
        self.neuralese_len      = neuralese_len 
        self.lang_position      = PositionalEncoding(encoder_kwargs['hidden_size'], dropout=0.1, max_len=80)
        self.decoder_option     = decoder_option
        self.token_entropy_beta = token_entropy_beta
        self.share_embedding    = decoder_kwargs['share_embedding']
        self.use_end_token      = decoder_kwargs['use_end_token']
        if self.use_end_token:
            assert(gumbel_hard == True)
        if decoder_option == 'rnn':
            self.decoder = DecoderRNN(**decoder_kwargs)
        elif decoder_option == 'transformer' or decoder_option == 'transformer-parallel':
            self.decoder = TranslatorTransformer(**decoder_kwargs)
            self.word_layer = nn.Linear(decoder_kwargs['vocab_size'], decoder_kwargs['hidden_size'], bias=False)
            self.word_init  = nn.Linear(decoder_kwargs['hidden_size'], decoder_kwargs['vocab_size'])
            self.ctx_layer  = nn.Linear(decoder_kwargs['hidden_size'], decoder_kwargs['hidden_size'])
            self.position_layer = nn.Linear(1, decoder_kwargs['hidden_size'] // 2)
            self.h_layer = nn.Linear(decoder_kwargs['hidden_size'], decoder_kwargs['hidden_size'] // 2)
        print('{} Neuralese length: '.format(self.encoder_type), neuralese_len)
        print('{} Neuralese vocab size: '.format(self.encoder_type), self.decoder.vocab_size)
        print('{} entropy beta: '.format(self.encoder_type), token_entropy_beta)

        self.entropy     = Entropy()
        self.tau0        = 1.0
        self.ANNEAL_RATE = 0.00003
        self.MIN_TEMP    = 0.5
        self.hard        = gumbel_hard

    def update_tau(self, iteration):
        self.tau = np.maximum(self.tau0 * np.exp(-self.ANNEAL_RATE * iteration), self.MIN_TEMP)

    def _decode_rnn(self, h_t, c_t, ctx, ctx_mask, provided_pred_seq=None):
        # rollout translation
        pred_seq = []
        logits = []
        pred_seq_lengths = torch.zeros(ctx.shape[0]).to(self.device)
        w_t = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)

        entropy_total = 0
        prob = torch.ones(ctx.shape[0]).to(self.device)
        end_word = torch.zeros(ctx.shape[0], 1).to(self.device)
        for i in range(0, self.neuralese_len):
          if self.use_end_token:
            end_word = ((end_word + w_t.detach()[:,end_token_idx].unsqueeze(-1) == 1) > 0).float()
          h_t, c_t, alpha, logit_t   = self.decoder(w_t, h_t, c_t, ctx, ctx_mask)
          # If we are given a pred_seq to calculate probability for, use that.
          if provided_pred_seq is not None:
            w_t = provided_pred_seq[:,i,:]
          else: 
            w_t     = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)
          w_t = w_t * (1 - end_word).expand(w_t.shape)
          entropy_total += self.entropy(logit_t * (1 - end_word).expand(logit_t.shape))
          pred_seq += [w_t.unsqueeze(1)]
          logits   += [logit_t.unsqueeze(-1)]
          pred_seq_lengths = pred_seq_lengths + (1 - end_word[:,0])
          # Calculate probability
          pis = calculate_pis(logit_t)
          prob[end_word[:,0]==0]     *= torch.sum(pis * w_t, dim=-1)[end_word[:,0]==0]

        pred_seq = torch.cat(pred_seq, dim=1)  
        logits   = torch.cat(logits, dim=-1)
        if self.encoder_type == 'backtranslator':
            return logits, pred_seq_lengths.long().tolist(), entropy_total, prob
        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, prob

    def _decode_transformer(self, ctx, ctx_mask, h_init=None, provided_pred_seq=None):
        assert(self.use_end_token==False)
        pred_seq = []
        pred_seq_lengths = torch.zeros(ctx.shape[0]).to(self.device)
        if h_init is None:
          w_t = torch.zeros(ctx.shape[0], self.decoder.vocab_size).to(self.device)
        else:
          w_t = self.word_init(h_init)
        w_t_mask  = torch.ones(ctx.shape[0], 1).byte().to(self.device)

        ctx = self.ctx_layer(ctx)
        ctx = ctx * ctx_mask.float().unsqueeze(-1).expand(ctx.shape)
        entropy_total = 0
        share_embedding = self.share_embedding
        prob = torch.ones(ctx.shape[0]).to(self.device)
        for i in range(0, self.neuralese_len):
          # update ctx and ctx mask
          w_t_embedding = self.word_layer(w_t)
          ctx      = torch.cat([ctx, w_t_embedding.unsqueeze(1)], dim=1)
          ctx_mask = torch.cat([ctx_mask, w_t_mask], dim=1)

          # pass through transformer
          logit_t = self.decoder(ctx, ctx_mask)
          entropy_total += self.entropy(logit_t)

          # sample words from distribution
          if provided_pred_seq is not None:
            w_t = provided_pred_seq[:,i,:]
          else: 
            w_t       = F.gumbel_softmax(logit_t, hard=self.hard, tau=self.tau)

          #w_t       = F.softmax(logit_t, dim=-1)
          w_t_mask  = torch.ones(ctx.shape[0], 1).byte().to(self.device)
          # update outputs
          if not share_embedding:
            pred_seq += [w_t.unsqueeze(1)]
          else:
            pred_seq += [self.word_layer(w_t).unsqueeze(1)]
          pred_seq_lengths = pred_seq_lengths + 1
          # Calculate probability
          pis     = calculate_pis(logit_t)
          prob     *= torch.sum(pis * w_t, dim=-1)

        pred_seq = torch.cat(pred_seq, dim=1)  

        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, prob

    def _decode_transformer_parallel(self, ctx, ctx_mask, h_init=None, provided_pred_seq=None):
        assert(self.use_end_token==False)
        pred_seq = []
        pred_seq_lengths = self.neuralese_len * torch.ones(ctx.shape[0]).to(self.device)
        word_mask = torch.ones(ctx.shape[0], self.neuralese_len).byte().to(self.device)
        positions = (torch.arange(0, self.neuralese_len).to(self.device)).float()
        positions = positions.unsqueeze(-1).unsqueeze(0).expand(ctx.shape[0], self.neuralese_len, -1)
        positions = self.position_layer(positions)
        h_init    = self.h_layer(h_init).unsqueeze(1).expand(ctx.shape[0], self.neuralese_len, -1)
        word_init = torch.cat([positions, h_init], dim=-1)

        mask = torch.cat([ctx_mask, word_mask], dim=1)

        h = self.decoder(torch.cat([ctx, word_init], dim=1), mask, return_full=True)

        logits = h[:,-self.neuralese_len:,:]

        entropy_total = self.entropy(logits.contiguous().view(-1, logits.shape[-1])) * self.neuralese_len
        if provided_pred_seq is not None:
          pred_seq = provided_pred_seq
        else: 
          pred_seq = F.gumbel_softmax(logits.contiguous().view(-1, logits.shape[-1]), hard=self.hard, tau=self.tau)
          pred_seq = pred_seq.view(logits.shape)
        #pred_seq = logits

        # Calculate probabilities from logits and chosen values
        prob = torch.ones(ctx.shape[0]).to(self.device)
        for t in range(logits.size()[1]):
            logit_t = logits[:,t,:]
            pis     = calculate_pis(logit_t)
            w_t     = pred_seq[:,t,:]
            prob   *= torch.sum(pis * w_t, dim=-1)
                
        return pred_seq, pred_seq_lengths.long().tolist(), entropy_total, prob

    # If this is the translator, enc_inputs is a dictionary with keys inputs, lengths
    # If this is the speaker, enc_inputs is a dictionary with keys all_img_feats, all_action_embeddings
    def forward(self, enc_inputs, src_lang=None, provided_pred_seq=None):
        ctx, h_t, c_t, ctx_mask = self.encoder(**enc_inputs)
        #ctx = self.lang_position(ctx)
        ctx_mask = (ctx_mask == 1).byte()

        # rollout translation
        if self.decoder_option == 'rnn':
          pred_seq, pred_seq_lengths, entropy_total, prob = self._decode_rnn(h_t, c_t, ctx, ctx_mask, provided_pred_seq=provided_pred_seq)
        elif self.decoder_option == 'transformer':
          pred_seq, pred_seq_lengths, entropy_total, prob = self._decode_transformer(ctx, ctx_mask, h_t, provided_pred_seq=provided_pred_seq)
        elif self.decoder_option == 'transformer-parallel':
          pred_seq, pred_seq_lengths, entropy_total, prob = self._decode_transformer_parallel(ctx, ctx_mask, h_t, provided_pred_seq=provided_pred_seq)
        elif self.decoder_option == 'transformer_backtrans':
          raise ValueError('Not implemented.')

        if self.encoder_type == 'backtranslator':
            logits = pred_seq
            return pred_seq
        return pred_seq, pred_seq_lengths, - self.token_entropy_beta * entropy_total, prob


class CoGrounding(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(CoGrounding, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        proj_navigable_kwargs = OrderedDict([
            ('input_dim', img_feat_input_dim),
            ('hidden_dims', img_fc_dim),
            ('use_batchnorm', img_fc_use_batchnorm),
            ('dropout', img_dropout),
            ('fc_bias', fc_bias),
            ('relu', opts.mlp_relu)
        ])
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])

        self.num_predefined_action = 1

    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
                navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size

        pre_feat: previous attended feature, batch x feature_size

        question: this should be a single vector representing instruction

        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, navigable_mask

