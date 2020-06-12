import argparse
import numpy as np

import torch

from collections import OrderedDict
from env import R2RPanoBatch, R2RDataLoader, load_features
from eval import Evaluation
from utils import setup, read_vocab, Tokenizer, set_tb_logger, is_experiment, padding_idx, end_token_idx, resume_training, save_checkpoint, path_len_prob_random_train_R2R, path_len_prob_random_val_R2R, path_len_prob_random_train_R4R, path_len_prob_random_val_R4R, path_len_prob_R2R, path_len_prob_R4R
#from trainer import PanoSeq2SeqTrainer
from trainer_adsl import ADSLSeq2SeqTrainer
from agents import PanoSeq2SeqAgent
from models import EncoderRNN, CoGrounding, SpeakerFollowerBaseline, EnvDropFollower
from models.adsl_model import Speaker, NeuraleseGenerator
from models import Translator


parser = argparse.ArgumentParser(description='PyTorch for Matterport3D Agent with panoramic view and action')
# General options
parser.add_argument('--exp_name', default='experiments_', type=str,
                    help='name of the experiment. \
                        It decides where to store samples and models')
parser.add_argument('--exp_name_secondary', default='', type=str,
                    help='name of the experiment. \
                        It decides where to store samples and models')

parser.add_argument('--train_vocab',
                    default='tasks/R2R-pano/data/train_vocab.txt',
                    type=str, help='path to training vocab')
parser.add_argument('--trainval_vocab',
                    default='tasks/R2R-pano/data/trainval_vocab.txt',
                    type=str, help='path to training and validation vocab')
parser.add_argument('--img_feat_dir',
                    default='img_features/ResNet-152-imagenet.tsv',
                    type=str, help='path to pre-cached image features')
parser.add_argument('--dataset_name',
                    default='R2R',
                    type=str, help="Name of dataset. ['R2R', 'R4R']")

# Training options
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--train_iters_epoch', default=200, type=int,
                    help='number of iterations per epoch')
parser.add_argument('--max_num_epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--eval_every_epochs', default=1, type=int,
                    help='how often do we eval the trained model')
parser.add_argument('--patience', default=3, type=int,
                    help='Number of epochs with no improvement after which learning rate will be reduced.')
parser.add_argument('--min_lr', default=1e-6, type=float,
                    help='A lower bound on the learning rate of all param groups or each group respectively')
parser.add_argument('--seed', default=1, type=int,
                    help='random seed')
parser.add_argument('--train_data_augmentation', default=0, type=int,
                    help='Training with the synthetic data generated with speaker')
parser.add_argument('--epochs_data_augmentation', default=5, type=int,
                    help='Number of epochs for training with data augmentation first')

# General model options
parser.add_argument('--arch', default='cogrounding', type=str,
		            help='options: cogrounding | speaker-baseline')
parser.add_argument('--max_navigable', default=16, type=int,
                    help='maximum number of navigable locations in the dataset is 15 \
                         we add one because the agent can decide to stay at its current location')
parser.add_argument('--use_ignore_index', default=1, type=int,
                    help='ignore target after agent has ended')
parser.add_argument('--use_language', default=1, type=int,
                    help='have the encoder provide language information')
parser.add_argument('--use_VE', default=0, type=int,
                    help='Flag on whether to encode image dimensions in another latent space.')
parser.add_argument('--ve_hidden_dims', default=(512, 512), type=int,
                    help='output dimensions of variation encoder for image.')
parser.add_argument('--ve_beta', default = 0.1, type=float,
                    help='beta for the variational regularization.')
parser.add_argument('--ve_prior_type', default = 0, type=str,
                    help='Flag to determine whether we want to learn the prior for variational encoding')
parser.add_argument('--ve_extra_input', default = '', type=str,
                    help='Input to pass in with visual feats into VIB')

# Agent options
parser.add_argument('--follow_gt_traj', default=0, type=int,
                    help='the shortest path to the goal may not match with the instruction if we use student forcing, '
                         'we provide option that the next ground truth viewpoint will try to steer back to the original'
                         'ground truth trajectory')
parser.add_argument('--teleporting', default=1, type=int,
                    help='teleporting: jump directly to next viewpoint. If not enabled, rotate and forward until you reach the '
                         'viewpoint with roughly the same heading')
parser.add_argument('--max_episode_len', default=10, type=int,
                    help='maximum length of episode')
parser.add_argument('--feedback_training', default='sample', type=str,
                    help='options: sample | mistake (this is the feedback for training only)')
parser.add_argument('--feedback', default='argmax', type=str,
                    help='options: sample | argmax (this is the feedback for testing only)')
parser.add_argument('--entropy_weight', default=0.0, type=float,
                    help='weighting for entropy loss')
parser.add_argument('--fix_action_ended', default=1, type=int,
                    help='Action set to 0 if ended. This prevent the model keep getting loss from logit after ended')

# Image context
parser.add_argument('--img_feat_input_dim', default=2176, type=int,
                    help='ResNet-152: 2048, if use angle, the input is 2176')
parser.add_argument('--img_fc_dim', default=(128,), nargs="+", type=int)
parser.add_argument('--img_fc_use_batchnorm', default=1, type=int)
parser.add_argument('--img_dropout', default=0.5, type=float)
parser.add_argument('--mlp_relu', default=1, type=int, help='Use ReLu in MLP module')
parser.add_argument('--img_fc_use_angle', default=1, type=int,
                    help='add relative heading and elevation angle into image feature')

# Language model
parser.add_argument('--remove_punctuation', default=0, type=int,
                    help='the original ''encode_sentence'' does not remove punctuation'
                         'we provide an option here.')
parser.add_argument('--reversed', default=1, type=int,
                    help='option for reversing the sentence during encoding')
parser.add_argument('--lang_embed', default='lstm', type=str, help='options: lstm ')
parser.add_argument('--word_embedding_size', default=256, type=int,
                    help='default embedding_size for language encoder')
parser.add_argument('--rnn_hidden_size', default=256, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--rnn_num_layers', default=1, type=int)
parser.add_argument('--rnn_dropout', default=0.5, type=float)
parser.add_argument('--max_cap_length', default=80, type=int, help='maximum length of captions')

# Evaluation options
parser.add_argument('--eval_only', default=0, type=int,
                    help='No training. Resume from a model and run evaluation')
parser.add_argument('--test_submission', default=0, type=int,
                    help='No training. Resume from a model and run testing for submission')
parser.add_argument('--eval_beam', default=0, type=int,
                    help='No training. Resume from a model and run with beam search')
parser.add_argument('--beam_size', default=5, type=int,
                    help='The number of beams used with beam search')

# Output options
parser.add_argument('--results_dir',
                    default='tasks/R2R-pano/results/',
                    type=str, help='where to save the output results for computing accuracy')
parser.add_argument('--resume', default='', type=str,
                    help='two options for resuming the model: latest | best')
parser.add_argument('--resume_base_dir', default='', type=str,
                    help='base directory for where pre-trained models are.')
parser.add_argument('--checkpoint_dir',
                    default='tasks/R2R-pano/checkpoints/pano-seq2seq/',
                    type=str, help='where to save trained models')
parser.add_argument('--tensorboard', default=1, type=int,
                    help='Use TensorBoard for loss visualization')
parser.add_argument('--log_dir',
                    default='tensorboard_logs/pano-seq2seq',
                    type=str, help='path to tensorboard log files')

# Neuralese options
parser.add_argument('--neuralese_vocab_size',
                    default=16,
                    type=int, help='neuralese vocabulary size')
parser.add_argument('--neuralese_len',
                    default=16,
                    type=int, help='neuralese max length')
parser.add_argument('--use_translator',
                    default=0,
                    type=int, help='whether to use translator')
parser.add_argument('--use_backtranslator',
                    default=0,
                    type=int, help='whether to use backtranslator')
parser.add_argument('--use_speaker',
                    default=0,
                    type=int, help='whether to use speaker')
parser.add_argument('--sampling_strategy',
                    default='shortest',
                    type=str, help='path sampling strategy for data augmentation')
parser.add_argument('--translator_encoder_rnn_dropout',
                    default=0.5,
                    type=float, help='translator encoder rnn dropout ratio')
parser.add_argument('--translator_decoder_arch',
                    default='transformer-parallel',
                    type=str, help='translator decoder options')
parser.add_argument('--translator_decoder_rnn_dropout',
                    default=0,
                    type=float, help='translator decoder rnn dropout ratio')
parser.add_argument('--translator_decoder_transformer_dropout',
                    default=0,
                    type=float, help='translator decoder transformer dropout ratio')
parser.add_argument('--translator_decoder_transformer_nheads',
                    default=4,
                    type=int, help='translator decoder transformer number of heads')
parser.add_argument('--translator_decoder_transformer_nlayers',
                    default=1,
                    type=int, help='translator decoder transformer number of layers')
parser.add_argument('--translator_entropy_beta',
                    default=0,
                    type=float, help='the entropy regularization beta over tokens')
parser.add_argument('--backtranslator_beta',
                    default=0.1,
                    type=float, help='regulurization beta for backtranslation')
parser.add_argument('--speaker_decoder_arch',
                    default='rnn',
                    type=str, help='speaker decoder options')
parser.add_argument('--speaker_decoder_rnn_dropout',
                    default=0,
                    type=float, help='speaker decoder rnn dropout ratio')
parser.add_argument('--speaker_decoder_transformer_dropout',
                    default=0,
                    type=float, help='speaker decoder transformer dropout ratio')
parser.add_argument('--speaker_decoder_transformer_nheads',
                    default=4,
                    type=int, help='speaker decoder transformer number of heads')
parser.add_argument('--speaker_decoder_transformer_nlayers',
                    default=1,
                    type=int, help='speaker decoder transformer number of layers')
parser.add_argument('--speaker_entropy_beta',
                    default=0,
                    type=float, help='the entropy regularization beta over tokens')
parser.add_argument('--kl_pq_beta',
                    default=0.1, 
                    type=float, help='the KL div(tran|speak) regularization beta')
parser.add_argument('--kl_qp_beta',
                    default=0.1, 
                    type=float, help='the KL div(speak|tran) regularization beta')
parser.add_argument('--share_embedding',
                    default=0,
                    type=int, help='whether to share neuralese embedding')
parser.add_argument('--neuralese_student_forcing',
                    default=0,
                    type=int, help='whether to use student forcing in neuralese generation')

# Neuralese Training Options 
parser.add_argument('--use_val_loader',
                    default=0,
                    type=int, help='whether to use loader for speaker evaluation')
parser.add_argument('--use_val_env',
                    default=0,
                    type=int, help='whether to use env for navigation evaluation')
parser.add_argument('--use_train_loader',
                    default=0,
                    type=int, help='whether to use R2R env for speaker training, mainly for consistency for debugging')
parser.add_argument('--sample_max_path_len',
                    default=-1,
                    type=int, help='control the path len mask')
parser.add_argument('--sample_fix_path_len',
                    default=-1,
                    type=int, help='control the path len mask')
parser.add_argument('--training_mode',
                    default='self-training',
                    type=str, help='choices of training modes - (self-training, self-training-R2R, trans-training, joint)')
parser.add_argument('--evaluation_mode',
                    default='self-eval',
                    type=str, help='choices of eval modes - (self-eval, R2R)')
parser.add_argument('--curriculum_learning', 
                    default=0, 
                    type=int, help='whether to slowly increase possible path length during self-training')
parser.add_argument('--compositional_training', 
                    default=0,
                    type=int, help='whether to use compositional training')
parser.add_argument('--use_end_token', 
                    default=0, 
                    type=int, help='whether to use END token for neuralese')
parser.add_argument('--use_teacher_forcing',
                    default=0,
                    type=int, help='use teacher forcing during training of the speaker')
parser.add_argument('--gumbel_hard',
                    default=1,
                    type=int, help='whether to use straight-through sampling on Gumbel-Softmax')
parser.add_argument('--speaker_var_length_neuralese',
                    default=0,
                    type=int, help='whether to use straight-through sampling on Gumbel-Softmax')
parser.add_argument('--joint_pre_training',
                    default=0,
                    type=int, help='how many epochs of pre-training on self-supervised trajectories')
parser.add_argument('--joint_translator_every_k',
                    default=1,
                    type=int, help='train translator in every k iterations')
parser.add_argument('--use_nmt_loss',
                    default=0,
                    type=int, help='Use machine translation loss as signal to train translator instead of navi loss')
parser.add_argument('--freeze_during_translator',
                    default=0,
                    type=int, help='Only update translator weights during a translation step')
parser.add_argument('--cyclic_reconstruction_only',
                    default=0,
                    type=int, help='When training translation, only use signal from translations and not navigation')
parser.add_argument('--ssl_beta',
                    default=1.0,
                    type=float, help='Coefficient for self-supervised learning loss')
parser.add_argument('--translation_beta',
                    default=1.0,
                    type=float, help='Coefficient for translation path loss')
parser.add_argument('--no_vision_feats',
                    default=0,
                    type=int, help='Flip flag to 1 to not use vision features during self-training')
parser.add_argument('--limited_vision_feats',
                    default=0,
                    type=int, help='Flip flag to 1 to only use 128 features for img_feats')
parser.add_argument('--no_language_feats',
                    default=0,
                    type=int, help='Flip flag to 1 to not use language features during training')
parser.add_argument('--replace_ctx_w_goal',
                    default=0,
                    type=int, help='Flip flag to 1 to replace lang ctx with goal img feats')
parser.add_argument('--only_keep_five',
                    default=0,
                    type=int, help='Flip flag to 1 to only use last five tokens in instruction')
parser.add_argument('--use_random_loader',
                    default=0,
                    type=int, help='whether to use ssl loader with non bm probabilities')
parser.add_argument('--max_node_in_path',
                    default=-1,
                    type=int, help='max_node_in_path')
parser.add_argument('--greedy_decode',
                    default=0,
                    type=int, help='whether to use greedy decoding for neuralese')
               
# Mimicking Speaker-Follower baseline
parser.add_argument('--translator_pre_training_epochs', default=200,
                    type=int, help='number of epochs for translator to establish neuralese language')
parser.add_argument('--speaker_pre_training_epochs', default=200, 
                    type=int, help='number of epochs for speaker to learn neuralese language')
parser.add_argument('--train_agent_speaker_epochs', default=100,
                    type=int, help='number of epochs for agent to train on augmented data from speaker')
parser.add_argument('--train_agent_translator_epochs', default=200,
                    type=int, help='number of epochs for agent to train on benchmark data from translator')

# Testing whether segmentation and compositional training works
parser.add_argument('--translator_segment', 
                    default=0,
                    type=int, help='whether to segment inputs to producing Neuralese')
parser.add_argument('--translator_segment_ctx', 
                    default=0,
                    type=int, help='whether to segment ctx when producing Neuralese')
parser.add_argument('--speaker_segment', 
                    default=0,
                    type=int, help='whether to segment inputs to producing Neuralese')
parser.add_argument('--speaker_segment_ctx', 
                    default=0,
                    type=int, help='whether to segment ctx when producing Neuralese')
parser.add_argument('--speaker_segment_prob',
                    default=0.2,
                    type=int, help='the probability of reset memory for generating compositional segments')
parser.add_argument('--num_neuralese_per_segment',
                    default=4,
                    type=int, help='How many tokens per "segment" (sentence of path segment)')
parser.add_argument('--max_num_sentences',
                    default=10,
                    type=int, help='Maximum number of sentences in any one instruction.') # Note: should actually find real value for this

# RL hyper parameters
parser.add_argument('--tf_pre_training', default=0,
                    type=int, help='number of epochs for agent to pre-train with teacher forcing')
parser.add_argument('--tf_decay_epochs', default=100,
                    type=int, help='number of epochs for agent to interleave tf and rl')
parser.add_argument('--tf_rate', default=0.5,
                    type=float, help='initial rate of sampling tf batches')

# Data augmentation using speaker follower
parser.add_argument('--use_SF_speaker',
                    default=0,
                    type=int, help='whether to use english speaker for data augmentation.')
parser.add_argument('--use_SF_angle', default=0, type=int,
                    help='whether to use speaker-followers angle feature representation')
parser.add_argument('--SF_speaker_model_prefix', default='tasks/R2R-pano/snapshots/release/speaker_final_release', type=str,
                    help='SF speaker model prefix')
parser.add_argument('--speaker_DA_strategy', default='pretrain', type=str,
                    help='SF speaker data augmentation strategy. mixed | pretrain')
parser.add_argument('--speaker_DA_epochs', default=200, type=int,
                    help='SF speaker data augmentation number of epochs.')

def get_encoder(opts, device, vocab):
    share_embedding = opts.share_embedding == 1
    encoder_kwargs = OrderedDict([
        ('opts', opts),
        ('vocab_size', opts.neuralese_vocab_size if opts.use_translator or opts.use_speaker else len(vocab)),
        ('embedding_size', opts.rnn_hidden_size if share_embedding else opts.word_embedding_size),
        ('hidden_size', opts.rnn_hidden_size),
        ('padding_idx', padding_idx),
        ('dropout_ratio', opts.rnn_dropout),
        ('bidirectional', opts.bidirectional == 1),
        ('num_layers', opts.rnn_num_layers),
        ('use_linear', True if opts.use_translator or opts.use_speaker else False),
        ('share_embedding', opts.share_embedding == 1),
    ])
    print('Using {} as encoder ...'.format(opts.lang_embed))
    if 'lstm' in opts.lang_embed:
        encoder = EncoderRNN(**encoder_kwargs)
    else:
        raise ValueError('Unknown {} language embedding'.format(opts.lang_embed))
    print(encoder)

    encoder = encoder.to(device)

    return encoder


def get_policy_model(opts, device):
    opts.env_drop_follower = False
    policy_model_kwargs = OrderedDict([
        ('opts', opts),
        ('img_fc_dim', opts.img_fc_dim),
        ('img_fc_use_batchnorm', opts.img_fc_use_batchnorm == 1),
        ('img_dropout', opts.img_dropout),
        ('img_feat_input_dim', opts.img_feat_input_dim),
        ('rnn_hidden_size', opts.rnn_hidden_size),
        ('rnn_dropout', opts.rnn_dropout),
        ('max_len', opts.max_cap_length),
        ('max_navigable', opts.max_navigable),
        ('use_VE', opts.use_VE)
    ])

    if opts.arch == 'cogrounding':
        model = CoGrounding(**policy_model_kwargs)
    elif opts.arch == 'speaker-baseline':
        model = SpeakerFollowerBaseline(**policy_model_kwargs)
    elif opts.arch == 'env-drop-follower':
        model = EnvDropFollower(**policy_model_kwargs)
        opts.env_drop_follower = True
    else:
        raise ValueError('Unknown {} model for seq2seq agent'.format(opts.arch))
    print(model)

    model = model.to(device)

    return model


def get_backtranslator(opts, device, vocab):
    if opts.use_backtranslator:
        backtranslator_encoder_kwargs = OrderedDict([
            ('vocab_size', opts.neuralese_vocab_size),
            ('embedding_size', opts.word_embedding_size),
            ('hidden_size', opts.rnn_hidden_size),
            ('padding_idx', None), 
            ('dropout_ratio', opts.translator_encoder_rnn_dropout),
            ('bidirectional', opts.bidirectional == 1),
            ('num_layers', opts.rnn_num_layers),
            ('use_linear', True),
        ])
        if opts.translator_decoder_arch == 'rnn':
            backtranslator_decoder_kwargs = OrderedDict([
                ('vocab_size', len(vocab)),
                ('vocab_embedding_size', opts.word_embedding_size),
                ('hidden_size', opts.rnn_hidden_size),
                ('dropout_ratio', opts.translator_decoder_rnn_dropout),
                ('use_linear', True), 
                ('share_embedding', opts.share_embedding==1),
                ('use_end_token', opts.use_end_token==1),
            ])
        else:
            raise NotImplementedError            
        backtranslator = Translator(backtranslator_encoder_kwargs, \
                                    backtranslator_decoder_kwargs, \
                                    neuralese_len=opts.max_cap_length, \
                                    gumbel_hard=1, \
                                    decoder_option=opts.translator_decoder_arch, \
                                    token_entropy_beta=opts.translator_entropy_beta, \
                                    is_backtranslator=True)
        backtranslator = backtranslator.to(device)
    else:
        backtranslator = None
    print(backtranslator)
    return backtranslator 

def get_translator(opts, device, vocab):
    if opts.use_translator:
        translator_encoder_kwargs = OrderedDict([
            ('vocab_size', len(vocab)),
            ('embedding_size', opts.word_embedding_size),
            ('hidden_size', opts.rnn_hidden_size),
            ('padding_idx', padding_idx),
            ('dropout_ratio', opts.translator_encoder_rnn_dropout),
            ('bidirectional', opts.bidirectional == 1),
            ('num_layers', opts.rnn_num_layers),
        ])
        if opts.translator_decoder_arch == 'rnn':
          translator_decoder_kwargs = OrderedDict([
              ('vocab_size', opts.neuralese_vocab_size),
              ('vocab_embedding_size', opts.word_embedding_size),
              ('hidden_size', opts.rnn_hidden_size),
              ('dropout_ratio', opts.translator_decoder_rnn_dropout),
              ('use_linear', True),
              ('share_embedding', opts.share_embedding==1),
              ('use_end_token', opts.use_end_token==1),
          ])
        elif opts.translator_decoder_arch == 'transformer' or opts.translator_decoder_arch == 'transformer-parallel':
          translator_decoder_kwargs = OrderedDict([
              ('vocab_size', opts.neuralese_vocab_size),
              ('hidden_size', opts.rnn_hidden_size),
              ('dropout_ratio', opts.translator_decoder_transformer_dropout),
              ('num_heads', opts.translator_decoder_transformer_nheads),
              ('num_layers', opts.translator_decoder_transformer_nlayers),
              ('share_embedding', opts.share_embedding==1),
              ('use_end_token', opts.use_end_token==1),
          ])
        elif opts.translator_decoder_arch == 'tcn':
          translator_decoder_kwargs = OrderedDict([
              ('vocab_size', opts.neuralese_vocab_size),
              ('hidden_size', opts.rnn_hidden_size),
              ('share_embedding', opts.share_embedding==1),
              ('use_end_token', opts.use_end_token==1),
          ])
        elif opts.translator_decoder_arch == 'tcn_skip':
          translator_decoder_kwargs = OrderedDict([
              ('vocab_size', opts.neuralese_vocab_size),
              ('hidden_size', opts.rnn_hidden_size),
              ('share_embedding', opts.share_embedding==1),
              ('use_end_token', opts.use_end_token==1),
          ])
        else:
          raise ValueError('Not implemented')
   
        translator = Translator(translator_encoder_kwargs, \
                                translator_decoder_kwargs, \
                                neuralese_len=opts.neuralese_len, \
                                gumbel_hard=opts.gumbel_hard, \
                                decoder_option=opts.translator_decoder_arch, \
                                token_entropy_beta=opts.translator_entropy_beta,
                                segment_by_sent=opts.translator_segment,
                                segment_ctx = opts.translator_segment_ctx,
                                num_neuralese = opts.num_neuralese_per_segment,
                                max_num_sentences = opts.max_num_sentences,
                                greedy_decode=opts.greedy_decode,
                                )
        translator = translator.to(device)
    else:
        translator = None

    print(translator)

    return translator


def get_speaker(opts, max_path_length, device):
    if opts.use_speaker:
        if opts.dataset_name == 'R4R':
            max_path_length = max_path_length * 2 + 4 # extra 4 is buffer for nodes that connect path A to B
        speaker_encoder_kwargs = OrderedDict([
            ('max_node_in_path',   max_path_length + 1),
            ('neuralese_len',      opts.neuralese_len),
            ('img_feat_input_dim', opts.img_feat_input_dim),
            ('hidden_size',         opts.rnn_hidden_size),
            ('dropout_ratio',      0.5)
        ]) 
        if opts.speaker_decoder_arch == 'rnn':
          speaker_decoder_kwargs = OrderedDict([
              ('vocab_size', opts.neuralese_vocab_size),
              ('vocab_embedding_size', opts.word_embedding_size),
              ('hidden_size', opts.rnn_hidden_size),
              ('dropout_ratio', opts.speaker_decoder_rnn_dropout),
              ('use_linear', True),
              ('share_embedding', opts.share_embedding==1),
          ])
        elif opts.speaker_decoder_arch == 'transformer' or opts.speaker_decoder_arch == 'transformer-parallel':
          raise NotImplementedError
          speaker_decoder_kwargs = OrderedDict([
              ('vocab_size', opts.neuralese_vocab_size),
              ('hidden_size', opts.rnn_hidden_size),
              ('dropout_ratio', opts.speaker_decoder_transformer_dropout),
              ('num_heads', opts.speaker_decoder_transformer_nheads),
              ('num_layers', opts.speaker_decoder_transformer_nlayers),
              ('share_embedding', opts.share_embedding==1),
              ('use_end_token', opts.use_end_token==1),
          ])
        else:
          raise ValueError('Not implemented')
   
        speaker = Speaker(speaker_encoder_kwargs,
                          speaker_decoder_kwargs,
                          opts.neuralese_len,
                          gumbel_hard=opts.gumbel_hard,
                          token_entropy_beta=opts.speaker_entropy_beta,
                          var_length_neuralese=opts.speaker_var_length_neuralese,
                          compositional=False, # set to False for now, mainly using compositional_by_reset
                          segment=opts.speaker_segment,
                          segment_ctx=opts.speaker_segment_ctx,
                          segment_prob=opts.speaker_segment_prob,
                          num_neuralese=opts.num_neuralese_per_segment,
                          max_num_sentences=opts.max_num_sentences)

        #speaker = NeuraleseGenerator('speaker',
        #                             speaker_encoder_kwargs, \
        #                             speaker_decoder_kwargs, \
        #                             neuralese_len=opts.neuralese_len, \
        #                             gumbel_hard=opts.gumbel_hard, \
        #                             decoder_option=opts.speaker_decoder_arch, \
        #                             token_entropy_beta=opts.speaker_entropy_beta)
                            
        speaker = speaker.to(device)
    else:
        speaker = None

    print(speaker)

    return speaker


def get_SF_speaker(opts, loader, device):
    import SF_utils.initialize_speaker as initialize_speaker
    speaker_model_prefix = opts.SF_speaker_model_prefix
    if opts.dataset_name == 'R2R':
        speaker, loader = initialize_speaker.entry_point(loader, speaker_model_prefix)
    else:
        assert(1==0)

    print(speaker)

    return speaker, loader


def main(opts):

    if not opts.use_random_loader:
      if opts.dataset_name == 'R2R':
          path_len_prob_train = path_len_prob_R2R
      elif opts.dataset_name == 'R4R':
          path_len_prob_train = path_len_prob_R4R
      else:
          assert(1==0)
    else:
      if opts.dataset_name == 'R2R':
          path_len_prob_train = path_len_prob_random_train_R2R
      elif opts.dataset_name == 'R4R':
          path_len_prob_train = path_len_prob_random_train_R4R
      else:
          assert(1==0)

    if opts.dataset_name == 'R2R':
        path_len_prob_val = path_len_prob_R2R
    elif opts.dataset_name == 'R4R':
        path_len_prob_val = path_len_prob_R4R
    else:
        assert(1==0)
    if opts.replace_ctx_w_goal:
        opts.rnn_hidden_size = opts.img_feat_input_dim
    #if opts.follow_gt_traj:
    #    opts.max_episode_len *= 2
    opts.max_node_in_path = len(path_len_prob_train) + 1
    print('train loader path_len_prob (unnormalized): ', path_len_prob_train)
    print('val loader path_len_prob (unnormalized): ', path_len_prob_val)
    # set manual_seed and build vocab
    setup(opts, opts.seed)
    if opts.cyclic_reconstruction_only: # if we are only looking at cyclic reconstruction, we don't eval
        opts.eval_every_epochs = opts.max_num_epochs + 1
    # set opts based on training mode
 
    print('Traning mode: ', opts.training_mode)
    ctx_flag = False
    if '_ctx' in opts.training_mode:
        ctx_flag = True
        opts.training_mode = opts.training_mode[:-4]
        print('Training mode strip to: ', opts.training_mode)

    if '-speaker_compositional' in opts.training_mode:
        opts.training_mode = opts.training_mode[:-22]
        print('Training mode strip to: ', opts.training_mode)
        opts.translator_segment = 0
        opts.translator_segment_ctx = 0
        opts.speaker_segment = 1
        if ctx_flag:
          opts.speaker_segment_ctx = 1
        opts.neuralese_len = opts.num_neuralese_per_segment * opts.max_num_sentences

    if '-translator_compositional' in opts.training_mode:
        opts.training_mode = opts.training_mode[:-25]
        print('Training mode strip to: ', opts.training_mode)
        opts.translator_segment = 1
        opts.speaker_segment = 0
        opts.speaker_segment_ctx = 0
        if ctx_flag:
          opts.translator_segment_ctx = 1
        opts.neuralese_len = opts.num_neuralese_per_segment * opts.max_num_sentences

    if opts.training_mode == 'self-training':
        opts.use_speaker    = 1
        opts.use_translator = 0
        opts.use_backtranslator = 0
        opts.use_train_loader = 1
    elif opts.training_mode == 'self-training-R2R':
        opts.use_speaker    = 1
        opts.use_translator = 0
        opts.use_backtranslator = 0
        opts.use_train_loader = 0
    elif opts.training_mode == 'trans-training':
        opts.use_speaker    = 0
        opts.use_translator = 1
        opts.use_train_loader = 0
    elif opts.training_mode == 'joint' or opts.training_mode == 'joint-alternation' \
           or opts.training_mode == 'joint-vae-like' or opts.training_mode == 'joint-teacher':
        opts.use_speaker    = 1
        opts.use_translator = 1
        opts.use_backtranslator = 0
        opts.use_train_loader = 1
    elif opts.training_mode == 'joint-backtranslate' or opts.training_mode == 'joint-backtranslate-teacher':
        opts.use_speaker    = 1
        opts.use_translator = 1
        opts.use_backtranslator = 1
        opts.use_train_loader = 1
    elif opts.training_mode == 'joint-reinforce':
        opts.use_speaker    = 1
        opts.use_translator = 1
        opts.use_backtranslator = 0
        opts.use_train_loader = 1
    elif opts.training_mode == 'joint-speaker_DA' or \
         opts.training_mode == 'joint-speaker_DA-teacher' or \
         opts.training_mode == 'joint-speaker_DA-teacher-student':
        opts.max_num_epochs      = opts.max_num_epochs + opts.speaker_DA_epochs
        opts.use_speaker         = 0
        opts.use_SF_speaker      = 1
        opts.use_translator      = 0
        opts.use_backtranslator  = 0
        opts.use_train_loader    = 1
        opts.use_SF_angle        = 1
    elif opts.training_mode == 'joint-speaker_DA-rep':
        opts.max_num_epochs      = opts.max_num_epochs + opts.speaker_DA_epochs
        opts.use_speaker         = 0
        opts.use_SF_speaker      = 0
        opts.use_translator      = 0
        opts.use_backtranslator  = 0
        opts.use_train_loader    = 0
        opts.use_SF_angle        = 0
    elif opts.training_mode == 'standard':
        opts.use_speaker    = 0
        opts.use_translator = 0
        opts.use_backtranslator = 0
        opts.use_train_loader = 0
    elif opts.training_mode == 'standard-reinforce':
        opts.use_speaker    = 0
        opts.use_translator = 0
        opts.use_backtranslator = 0
        opts.use_train_loader = 0
    elif opts.training_mode == 'mimic-speaker-follower':
        opts.use_speaker = 1
        opts.use_translator = 1
        opts.use_train_loader = 1
        opts.max_num_epochs = opts.translator_pre_training_epochs \
                              + opts.speaker_pre_training_epochs \
                              + opts.train_agent_speaker_epochs \
                              + opts.train_agent_translator_epochs
    elif opts.training_mode == 'train-speaker':
        opts.use_speaker    = 1
        opts.use_translator = 0
        opts.use_backtranslator = 0
        opts.use_train_loader = 0
    else:
        raise ValueError('Wrong training mode.')

    if opts.evaluation_mode == 'self-eval-loader':
        opts.use_val_loader = 1
        opts.use_val_env    = 0
    elif opts.evaluation_mode == 'self-eval-R2R-env':
        opts.use_val_loader = 0
        opts.use_val_env    = 1
    elif opts.evaluation_mode == 'trans-eval':
        opts.use_val_loader = 0
        opts.use_val_env    = 1
    elif opts.evaluation_mode == 'standard-eval-env':
        opts.use_val_loader = 0
        opts.use_val_env    = 1
    elif opts.evaluation_mode == 'joint-eval' or opts.evaluation_mode == 'joint-speaker_DA-eval':
        opts.use_val_loader = 1
        opts.use_val_env    = 1
    elif opts.evaluation_mode == 'eval-speaker':
        opts.use_val_loader = 0
        opts.use_val_env    = 1
    else:
        raise ValueError('Wrong evaluation mode.')

    # joint training configurations
    if opts.joint_translator_every_k > 1:
        opts.max_num_epochs = opts.max_num_epochs * opts.joint_translator_every_k
    if opts.joint_pre_training > 0:
        opts.max_num_epochs = opts.max_num_epochs + opts.joint_pre_training 
    if opts.tf_pre_training > 0:
        opts.max_num_epochs = opts.max_num_epochs + opts.tf_pre_training 
    if opts.compositional_training:
        opts.speaker_var_length_neuralese = 1

    print(opts)
    max_path_length = np.where(np.array(path_len_prob_train) != 0)[0][-1] + 1
    if opts.speaker_var_length_neuralese and opts.neuralese_len < max_path_length * 2:
        assert ValueError('Max neuralese length must be at least 2 * max path length')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a batch training environment that will also preprocess text
    vocab = read_vocab(opts.train_vocab)
    tok = Tokenizer(opts.remove_punctuation == 1, opts.reversed == 1, vocab=vocab, encoding_length=opts.max_cap_length)

    # create language instruction encoder
    encoder = get_encoder(opts, device, vocab)

    # create policy model
    model   = get_policy_model(opts, device)

    # create translator and backtranslator model
    translator = get_translator(opts, device, vocab)
    backtranslator = get_backtranslator(opts, device, vocab)

    # create speeaker model
    if not opts.use_SF_speaker:
        speaker = get_speaker(opts, max_path_length, device)

    # get optimizer list
    optimizers = OrderedDict()
    agent_params        = list(encoder.parameters()) + list(model.parameters())
    optimizers['agent'] = torch.optim.Adam(agent_params, lr=opts.learning_rate)
    if opts.use_translator:
        translator_params = list(translator.parameters())
        optimizers['translator'] = torch.optim.Adam(translator_params, lr=opts.learning_rate)
    if opts.use_backtranslator:
        backtranslator_params = list(backtranslator.parameters())
        optimizers['backtranslator'] = torch.optim.Adam(backtranslator_params, lr=opts.learning_rate)
    if opts.use_speaker:
        speaker_params = list(speaker.parameters())
        optimizers['speaker'] = torch.optim.Adam(speaker_params, lr=opts.learning_rate)

    # optionally resume from a checkpoint
    if opts.resume in ['latest', 'best']:
        #raise ValueError('Need to be checked.')
        model, encoder, translator, speaker, optimizers, best_success_rate_loader, best_success_rate_env = resume_training(opts, model, encoder, translator, speaker, optimizers)

    # if a secondary exp name is specified, this is useful when resuming from a previous saved
    # experiment and save to another experiment, e.g., pre-trained on synthetic data and fine-tune on real data
    if opts.exp_name_secondary:
        opts.exp_name += opts.exp_name_secondary

    feature, img_spec = load_features(opts.img_feat_dir)

    if opts.test_submission:
        assert opts.resume, 'The model was not resumed before running for submission.'
        test_env = ('test', (R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size,
                                 splits=['test'], tokenizer=tok), Evaluation(['test'], dataset_name=opts.dataset_name)))
        agent_kwargs = OrderedDict([
            ('opts', opts),
            ('env', test_env[1][0]),
            ('results_path', ""),
            ('encoder', encoder),
            ('model', model),
            ('feedback', opts.feedback)
        ])
        agent = PanoSeq2SeqAgent(**agent_kwargs)
        # setup trainer
        trainer = PanoSeq2SeqTrainer(opts, agent, optimizer)
        epoch = opts.start_epoch - 1
        trainer.eval(epoch, test_env)
        return

    # set up R2R environments
    if not opts.train_data_augmentation:
        train_env = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
                                 splits=['train'], tokenizer=tok, dataset_name=opts.dataset_name)
    else:
        train_env = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
                                 splits=['synthetic'], tokenizer=tok, dataset_name=opts.dataset_name)

    val_envs = OrderedDict([(split, (R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size,
                                     splits=[split], tokenizer=tok, dataset_name=opts.dataset_name), Evaluation([split], dataset_name=opts.dataset_name)))
                for split in ['val_seen', 'val_unseen']]
               )

    if opts.use_train_loader:
        if opts.sample_fix_path_len > 0:
          print('constraining the path len with fixed length:', opts.sample_fix_path_len)
          path_len_mask = np.zeros(len(path_len_prob_train)) 
          path_len_mask[opts.sample_fix_path_len - 1] = 1
        elif opts.sample_max_path_len > 0:
          print('constraining the path len to up to:', opts.sample_max_path_len)
          sample_max_path_len = opts.sample_max_path_len
          path_len_mask = np.zeros(len(path_len_prob_train)) 
          path_len_mask[:sample_max_path_len] = 1
        else:
          path_len_mask = None
        train_loader = R2RDataLoader(opts, feature, img_spec, path_len_prob_train, batch_size=opts.batch_size,
                                     seed=opts.seed, splits=['train'], tokenizer=tok, sampling_strategy=opts.sampling_strategy,
                                     path_len_mask=path_len_mask, dataset_name=opts.dataset_name)
    else:
        train_loader = None

    if opts.training_mode == 'joint-speaker_DA-rep':
        train_loader = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
                                  splits=['synthetic'], tokenizer=tok, dataset_name=opts.dataset_name)

    if opts.use_val_loader:
        if opts.sample_fix_path_len > 0:
          print('constraining the path len with fixed length:', opts.sample_fix_path_len)
          path_len_mask_val = np.zeros(len(path_len_prob_val))
          path_len_mask_val[opts.sample_fix_path_len - 1] = 1
        elif opts.sample_max_path_len > 0:
          print('constraining the path len to up to:', opts.sample_max_path_len)
          sample_max_path_len = opts.sample_max_path_len
          path_len_mask_val = np.zeros(len(path_len_prob_val))
          path_len_mask_val[:sample_max_path_len] = 1
        else:
          path_len_mask_val = None
        val_loaders = OrderedDict([(split, (R2RDataLoader(opts, feature, img_spec, path_len_prob_val, batch_size=opts.batch_size,
                                            seed=opts.seed, splits=[split], tokenizer=tok, sampling_strategy=opts.sampling_strategy,
                                            path_len_mask=path_len_mask_val, dataset_name=opts.dataset_name),
                                            Evaluation([split], dataset_name=opts.dataset_name,
                                                       online=False, non_shortest_path=('random' in opts.sampling_strategy))
                                           )
                                   )
                       for split in ['train', 'val_unseen']]
                      )
    else:
        val_loaders = None

    if opts.use_SF_speaker:
        speaker, train_loader = get_SF_speaker(opts, train_loader, device)

    # create agent
    agent_kwargs = OrderedDict([
        ('opts', opts),
        ('env', train_env),
        ('results_path', ""),
        ('encoder', encoder),
        ('model', model),
        ('translator', translator),
        ('backtranslator', backtranslator),
        ('speaker', speaker),
        ('feedback', opts.feedback),
    ])
    agent = PanoSeq2SeqAgent(**agent_kwargs)

    # setup trainer
    # trainer = PanoSeq2SeqTrainer(opts, agent, optimizer, opts.train_iters_epoch) # Progress Monitor trainer
    trainer = ADSLSeq2SeqTrainer(opts, agent, optimizers, opts.train_iters_epoch)

    # --- This needs to be checked
    if opts.eval_beam or opts.eval_only:
        success_rate = []
        for val_env in val_envs.items():
            success_rate.append(trainer.eval(opts.start_epoch - 1, val_env, tb_logger=None))
        return
    # ---

    # set up tensorboard logger
    # tb_logger = set_tb_logger(opts.log_dir, opts.exp_name, opts.resume)
    tb_logger = None

    best_success_rate_loader = best_success_rate_loader if opts.resume in ['latest', 'best'] else 0.0
    best_success_rate_env    = best_success_rate_env if opts.resume in ['latest', 'best'] else 0.0
    best_sr_seen_loader = 0.0
    best_sr_seen_env    = 0.0

    if opts.curriculum_learning and opts.use_train_loader:
        print('Training with curriculum learning regime')
        # Calculate number of epochs before increasing max length. 
        # This divides training into intervals. The 2 extra intervals is to train with max path length for longer.
        epoch_increase_max_len = int(opts.max_num_epochs / (len(path_len_prob_train) + 2))
    for epoch in range(opts.start_epoch, opts.max_num_epochs + 1):
        if opts.curriculum_learning and opts.use_train_loader:
            path_len_mask = np.zeros(len(path_len_prob_train))
            # Start with path len of at most 2, and increase until we're at max path length.
            max_len = min(len(path_len_prob_train), int(epoch / epoch_increase_max_len) + 2)
            path_len_mask[:max_len] = 1
            train_loader.update_mask(path_len_mask)
        if opts.training_mode == 'mimic-speaker-follower' and epoch == opts.translator_pre_training_epochs + opts.speaker_pre_training_epochs + 1:
            # Reset the navigational portion of the agent, which includes the optimizer
            print('Reseting navigational agent')
            encoder = get_encoder(opts, device, vocab)
            model   = get_policy_model(opts, device)
            agent_params = list(encoder.parameters()) + list(model.parameters())
            trainer.agent.encoder = encoder
            trainer.agent.model   = model
            trainer.optimizers['agent'] = torch.optim.Adam(agent_params, lr=opts.learning_rate)

        trainer.train(epoch, train_env, train_loader, tb_logger, use_train_loader=opts.use_train_loader, mode=opts.training_mode)

        if epoch % opts.eval_every_epochs == 0:
            success_rate_loader = []
            success_rate_env    = []
            if opts.training_mode == 'mimic-speaker-follower': # Validate speaker and translator when relevant
                if epoch > opts.translator_pre_training_epochs and epoch <= opts.translator_pre_training_epochs + opts.speaker_pre_training_epochs:
                    opts.use_val_loader = True
                    opts.use_val_env = False
                else:
                    opts.use_val_loader = False
                    opts.use_val_env = True
            if opts.use_val_loader:
                for val_loader in val_loaders.items():
                    success_rate_loader.append(trainer.eval(epoch, None, val_loader, use_val_loader=True, tb_logger=tb_logger, evaluation_mode=opts.evaluation_mode))
                success_rate_compare_loader = success_rate_loader[1]
                sr_seen_compare_loader      = success_rate_loader[0]
            if opts.use_val_env:
                for val_env in val_envs.items():
                    success_rate_env.append(trainer.eval(epoch, val_env, None, use_val_loader=False, tb_logger=tb_logger, evaluation_mode=opts.evaluation_mode))
                success_rate_compare_env = success_rate_env[1]
                sr_seen_compare_env      = success_rate_env[0]

            if is_experiment():
                is_best_loader = None
                is_best_env = None
                # remember best val_seen success rate and save checkpoint
                if len(success_rate_loader) > 0:
                  is_best_loader = success_rate_compare_loader >= best_success_rate_loader
                  best_success_rate_loader = max(success_rate_compare_loader, best_success_rate_loader)
                  best_sr_seen_loader      = max(sr_seen_compare_loader,      best_sr_seen_loader)
                  print("--> Highest val_seen success rate loader: {}".format(best_sr_seen_loader))
                  print("--> Highest val_unseen success rate loader: {}".format(best_success_rate_loader))
                if len(success_rate_env) > 0:
                  is_best_env = success_rate_compare_env >= best_success_rate_env
                  best_success_rate_env = max(success_rate_compare_env, best_success_rate_env)
                  best_sr_seen_env = max(sr_seen_compare_env, best_sr_seen_env)
                  print("--> Highest val_seen success rate env: {}".format(best_sr_seen_env))
                  print("--> Highest val_unseen success rate env: {}".format(best_success_rate_env))

                # save the model if it is the best so far
                save_checkpoint(OrderedDict([
                    ('opts', opts),
                    ('epoch', epoch + 1),
                    ('state_dict', model.state_dict()),
                    ('encoder_state_dict', encoder.state_dict()),
                    ('translator_state_dict', translator.state_dict() if translator is not None else None),
                    ('speaker_state_dict', speaker.state_dict() if (speaker is not None) and (opts.use_SF_speaker==False) else None),
                    ('best_success_rate_loader', best_success_rate_loader),
                    ('best_success_rate_env', best_success_rate_env),
                    ('optimizers', optimizers),
                    ('max_episode_len', opts.max_episode_len),
                ]), is_best_loader == True or is_best_env == True, checkpoint_dir=opts.checkpoint_dir, name=opts.exp_name)

        # Progress monitoring code
        #if opts.train_data_augmentation and epoch == opts.epochs_data_augmentation:
        #    train_env = R2RPanoBatch(opts, feature, img_spec, batch_size=opts.batch_size, seed=opts.seed,
        #                             splits=['train'], tokenizer=tok)

    print("--> Finished training")


if __name__ == '__main__':
    opts = parser.parse_args()
    main(opts)
