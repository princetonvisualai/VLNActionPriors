import argparse

class arg_parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch for Matterport3D Agent with panoramic view and action')
        self._initialize(parser)

    def _initialize(self, parser):
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
        parser.add_argument('--use_speaker',
                            default=0,
                            type=int, help='whether to use speaker')
        parser.add_argument('--sampling_strategy',
                            default='shortest',
                            type=str, help='path sampling strategy for data augmentation')
        parser.add_argument('--translator_decoder_arch',
                            default='transformer-parallel',
                            type=str, help='translator decoder options')
        parser.add_argument('--translator_encoder_rnn_dropout',
                            default=0,
                            type=float, help='translator encoder rnn dropout ratio')
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
        parser.add_argument('--share_embedding',
                            default=0,
                            type=int, help='whether to share neuralese embedding')
        parser.add_argument('--use_val_loader',
                            default=1,
                            type=int, help='whether to use loader for speaker evaluation')
        parser.add_argument('--use_train_loader',
                            default=0,
                            type=int, help='whether to use R2R env for speaker training, mainly for consistency for debugging')
        parser.add_argument('--sample_max_path_len',
                            default=-1,
                            type=int, help='control the path len mask')
        parser.add_argument('--training_mode',
                            default='self-training',
                            type=str, help='choices of training modes - (self-training, natural-language-training, joint)')
        parser.add_argument('--evaluation_mode',
                            default='self-eval',
                            type=str, help='choices of eval modes - (self-eval, R2R)')
        self.args = parser.parse_args()


    def get_args(self):
        return self.args
