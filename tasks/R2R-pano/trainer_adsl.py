import time
import math
import numpy as np

from collections import OrderedDict
import torch
from utils import AverageMeter, load_datasets


"""
Trainer for neuralese driven navigation agent training and evaluation.

Training modes: 
     * neuralese training
     * translator training
     * join training (neuralese + translator)
     * To be added: RL model using MI as initialization
Eval modes: 
     * evaluation on self-generated trajectories
     * evaluation on standard benchmarks
"""
class ADSLSeq2SeqTrainer():
    """Trainer for training and validation process"""
    def __init__(self, opts, agent, optimizers, train_iters_epoch=100):
        self.opts  = opts
        self.agent = agent
        self.optimizers = optimizers
        self.train_iters_epoch   = train_iters_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_tf_rate = self.opts.tf_rate
        self.tf_rate = self.opts.tf_rate
        self.tf_decay_iters = self.opts.tf_decay_epochs * self.train_iters_epoch

    def update_tf_every_k(self, iter):
        self.tf_rate = max(0, self.init_tf_rate - self.init_tf_rate * (float(iter) / float(self.tf_decay_iters)))

    def backward_zero_grad(self, optimizers, keys=None):
        if keys is None:
          keys = optimizers.keys()
        for k in keys:
          optimizers[k].zero_grad()

    def backward_compute(self, optimizers, loss, keys=None):
        if keys is None:
          keys = optimizers.keys()
        loss.backward()
        for k in keys:
          optimizers[k].step()

    def train(self, epoch, train_env, train_loader, tb_logger=None, use_train_loader=False, mode=None):
        if mode == 'trans-training':
            assert(use_train_loader == False and train_env is not None)
        if mode == 'joint-alternation':
            assert(train_env is not None and train_loader is not None)
        if mode == 'joint' or mode == 'joint-backtranslate' or mode == 'joint-reinforce' or \
           mode == 'joint-teacher' or mode == 'joint-backtranslate-teacher' or \
           mode == 'joint-speaker_DA' or mode == 'joint-speaker_DA-teacher-student' or \
           mode == 'joint-speaker_DA-rep':
            assert(train_env is not None and train_loader is not None)
        if mode == 'self-training':
            assert(use_train_loader == True and train_loader is not None)
        if mode == 'self-training-R2R':
            assert(use_train_loader == False and train_env is not None)
        batch_time  = AverageMeter()
        navi_losses = AverageMeter() # The loss of path: natural language -> neuralese -> trajectories
        mi_losses   = AverageMeter()
        entropies   = AverageMeter()
        dists     = AverageMeter()
        movements = AverageMeter()

        val_navi_losses = AverageMeter() # The loss of path: natural language -> neuralese -> trajectories
        val_mi_losses   = AverageMeter()
        val_acces = AverageMeter()

        print('Training on {} env ...'.format(train_env.splits[0]))
        # switch to train mode
        if not use_train_loader:
            print('Using {} benchmark environment for training'.format(self.opts.dataset_name))
            self.agent.env = train_env
        else:
            print('Using self-supervised learning environment for training')
            self.agent.env = train_loader

        if self.agent.translator is not None:
            self.agent.translator.train()
        if self.agent.speaker is not None and not (mode == "joint-speaker_DA" or mode == "joint-speaker_DA-teacher" or mode == 'joint-speaker_DA-teacher-student'):
            self.agent.speaker.train()
        self.agent.encoder.train()
        self.agent.model.train()

        self.agent.feedback = self.opts.feedback_training
        self.agent.val_acc = None

        # load dataset path for computing ground truth distance
        # An unnecessary part of only training speaker+cogrounding, but harmless to leave in
        self.agent.gt = OrderedDict()
        for item in load_datasets(train_env.splits, self.opts, dataset_name=self.opts.dataset_name):
            self.agent.gt[item['path_id']] = item

        end = time.time()
        for iter in range(1, self.train_iters_epoch + 1):
            optim_keys    = None
            keep_grad     = False
            if mode == "self-training" or mode =="self-training-R2R":
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                mi_loss, traj = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, train=True)
                navi_loss = None
                loss = mi_loss
            elif mode == "trans-training":
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                navi_loss, traj = self.agent.rollout_cogrounding()
                mi_loss = None
                loss = navi_loss
            elif mode == "joint-alternation":
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                if iter % self.opts.joint_translator_every_k == 0 and epoch > self.opts.joint_pre_training:
                    # benchmark loader
                    self.agent.env = train_env
                    if self.opts.use_nmt_loss:
                        optim_keys    = ['translator']
                        # navi loss is actually nmt loss, but code is a bit cleaner this way
                        navi_loss, traj = self.agent.machine_translate() 
                    else:
                        navi_loss, traj = self.agent.rollout_cogrounding()
                        if self.opts.freeze_during_translator:
                            optim_keys = ['translator']
                    mi_loss = None
                    loss = navi_loss
                else:
                    # self-training loader
                    self.agent.env = train_loader
                    mi_loss, traj, = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, train=True)
                    navi_loss = None
                    loss = mi_loss
            elif mode == "joint" or mode == 'joint-teacher' or  mode == "joint-backtranslate" or mode == "joint-backtranslate-teacher":
                if mode == 'joint-teacher' or mode == "joint-backtranslate-teacher":
                    use_teacher_forcing = True
                else:
                    use_teacher_forcing = False
                self.backward_zero_grad(self.optimizers, keys=optim_keys)
                keep_grad = True
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                # self-training loader
                self.agent.env = train_loader
                mi_loss, _ = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, train=True, teacher_forcing=use_teacher_forcing)
                ssl_loss = self.opts.ssl_beta * mi_loss
                ssl_loss.backward()
                # benchmark loader
                self.agent.env = train_env
                navi_loss, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                loss = self.opts.translation_beta * navi_loss
            elif mode == "joint-speaker_DA-rep":
                # whether to use teacher forcing for ssl loop
                use_teacher_forcing = False
                mi_loss    = None
                navi_loss  = None
                use_loader = False
                if epoch < self.opts.speaker_DA_epochs:
                    # self-training loader
                    self.agent.env = train_loader
                    use_loader = True
                else:
                    # benchmark loader
                    self.agent.env = train_env
                navi_loss, traj = self.agent.rollout_cogrounding(use_loader=use_loader, train=True, teacher_forcing=False)
                loss = self.opts.translation_beta * navi_loss
            elif mode == "joint-speaker_DA" or mode == "joint-speaker_DA-teacher" or mode == "joint-speaker_DA-teacher-student":
                # whether to use teacher forcing for ssl loop
                if mode == "joint-speaker_DA-teacher":
                    use_teacher_forcing = True
                elif mode == "joint-speaker_DA-teacher-student":
                    use_teacher_forcing = True
                else:
                    use_teacher_forcing = False
                mi_loss = None
                navi_loss = None
                if self.opts.speaker_DA_strategy == 'pretrain':
                    if epoch < self.opts.speaker_DA_epochs:
                        self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                        # self-training loader
                        self.agent.env = train_loader
                        mi_loss, traj = self.agent.rollout_SF_speaker_cogrounding(use_loader=use_train_loader, train=True, teacher_forcing=use_teacher_forcing)
                        loss = self.opts.ssl_beta * mi_loss
                    else:
                        # benchmark loader
                        if mode == "joint-speaker_DA-teacher-student":
                            use_teacher_forcing = False
                        self.agent.env = train_env
                        navi_loss, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                        loss = self.opts.translation_beta * navi_loss

                elif self.opts.speaker_DA_strategy == 'mixed':
                    # get the anneal ratio
                    total_mixed = self.opts.speaker_DA_epochs
                    cur_iteration = iter + (epoch - 1) * self.train_iters_epoch
                    tot_iteration = total_mixed * self.train_iters_epoch
                    anneal_ratio  = 1 - (cur_iteration / tot_iteration)
                    ssl_beta_current = self.opts.ssl_beta * max(0, anneal_ratio)

                    if ssl_beta_current > 0:
                        # zero gradient and keep grads for first backward pass
                        self.backward_zero_grad(self.optimizers, keys=optim_keys)
                        keep_grad = True
                        self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)

                        # self-training loader
                        self.agent.env = train_loader
                        mi_loss, _ = self.agent.rollout_SF_speaker_cogrounding(use_loader=use_train_loader, train=True, teacher_forcing=use_teacher_forcing)
                        ssl_loss = ssl_beta_current * mi_loss
                        ssl_loss.backward()
                    else:
                        if mode == "joint-speaker_DA-teacher-student":
                            use_teacher_forcing = False

                    # benchmark loader
                    self.agent.env = train_env
                    navi_loss, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                    loss = self.opts.translation_beta * navi_loss

            elif mode == "joint-reinforce":
                self.backward_zero_grad(self.optimizers, keys=optim_keys)
                keep_grad = True
                self.update_tf_every_k(iter + (epoch - self.opts.tf_pre_training) * self.train_iters_epoch)
                coin = torch.rand(1)
                if coin.item() < self.tf_rate or epoch < self.opts.tf_pre_training:
                    use_teacher_forcing = True
                else:
                    use_teacher_forcing = False
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)

                # self-training loader
                self.agent.env = train_loader
                speaker_tf_loss, \
                speaker_traj = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, \
                                                                      train=True, teacher_forcing=use_teacher_forcing)
                speaker_rl_loss, speaker_reward = self.agent.get_reward_loss(speaker_traj, rav_type='speaker')
                if use_teacher_forcing:
                    mi_loss = speaker_tf_loss
                else:
                    mi_loss = speaker_rl_loss
                ssl_loss = self.opts.ssl_beta * mi_loss
                ssl_loss.backward()

                # benchmark loader
                self.agent.env = train_env
                navi_tf_loss, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                navi_rl_loss, navi_reward = self.agent.get_reward_loss(traj, rav_type='translator')
                if use_teacher_forcing:
                    navi_loss = navi_tf_loss
                else:
                    navi_loss = navi_rl_loss
                loss = self.opts.translation_beta * navi_loss
            elif mode == "joint-vae-like":
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                if epoch > self.opts.joint_pre_training:
                    # benchmark loader
                    self.agent.env = train_env
                    matching_loss = self.agent.rollout_cogrounding(vae_like=True)
                    matching_loss.backward()
                # self-training loader
                self.agent.env = train_loader
                mi_loss, traj, = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, train=True)
                navi_loss = None
                loss = mi_loss
            elif mode == "standard":
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                navi_loss, traj = self.agent.rollout_cogrounding()
                mi_loss = None
                loss = navi_loss
            elif mode == "standard-reinforce-mixed":
                self.update_tf_every_k(iter + (epoch - self.opts.tf_pre_training) * self.train_iters_epoch)
                coin = torch.rand(1)
                if coin.item() < self.tf_rate or epoch < self.opts.tf_pre_training:
                    use_teacher_forcing = True
                else:
                    use_teacher_forcing = False
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                navi_tf_loss, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                navi_rl_loss, navi_reward = self.agent.get_reward_loss(traj, rav_type='translator')
                if use_teacher_forcing:
                    navi_loss = navi_tf_loss
                else:
                    navi_loss = navi_rl_loss
                mi_loss = None
                loss = navi_loss
            elif mode == "standard-reinforce":
                use_teacher_forcing = False
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                if epoch < self.opts.tf_pre_training:
                    navi_loss, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                else:
                    _, traj = self.agent.rollout_cogrounding(teacher_forcing=use_teacher_forcing)
                    navi_rl_loss, navi_reward = self.agent.get_reward_loss(traj, rav_type='translator')
                    navi_loss = navi_rl_loss
                mi_loss = None
                loss = navi_loss
            elif mode == "mimic-speaker-follower":
                # Get which phase of mimicking speaker-follower we're in based on epoch
                phase_1_thresh = self.opts.translator_pre_training_epochs
                phase_2_thresh = phase_1_thresh + self.opts.speaker_pre_training_epochs
                phase_3_thresh = phase_2_thresh + self.opts.train_agent_speaker_epochs
                if epoch > phase_3_thresh: phase = 4
                if epoch > phase_2_thresh: phase = 3
                if epoch > phase_1_thresh: phase = 2
                else:                      phase = 1
                self.agent.update_cur_iteration(cur_iteration=iter + (epoch - 1) * self.train_iters_epoch)
                if phase == 1: # Learn neuralese: training translator and agent jointly
                    self.agent.env = train_env
                    navi_loss, traj = self.agent.rollout_cogrounding()
                    mi_loss = None
                    loss = self.opts.translation_beta * navi_loss
                if phase == 2: # Train speaker: fix agent while training speaker
                    optim_keys = ['speaker']
                    self.agent.env = train_loader
                    mi_loss, traj  = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, train=True)
                    navi_loss = None
                    loss = mi_loss
                if phase == 3: # Train agent on augmented data: Agent is reset in main file, freeze speaker.
                    optim_keys = ['agent']
                    self.agent.env = train_loader
                    mi_loss, traj  = self.agent.rollout_speaker_cogrounding(use_loader=use_train_loader, train=True)
                    navi_loss = None
                    loss = mi_loss
                if phase == 4: # Train agent on benchmark data: freeze translator
                    optim_keys = ['agent']
                    self.agent.env = train_env
                    navi_loss, traj = self.agent.rollout_cogrounding()
                    mi_loss = None
                    loss = self.opts.translation_beta * navi_loss
                    
            else:
                raise ValueError('Wrong training mode.')

            if traj is not None:
                dist_from_goal = np.mean(self.agent.dist_from_goal)
                movement = np.mean(self.agent.traj_length)

            if navi_loss is not None:
                navi_losses.update(navi_loss.item(), self.opts.batch_size)
            if mi_loss is not None:
                mi_losses.update(mi_loss.item(), self.opts.batch_size)
            if traj is not None:
                dists.update(dist_from_goal, self.opts.batch_size)
                movements.update(movement, self.opts.batch_size)

            # zero the gradients before backward pass
            if keep_grad == False:
                self.backward_zero_grad(self.optimizers, keys=optim_keys)
            self.backward_compute(self.optimizers, loss, keys=optim_keys)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if tb_logger and iter % 10 == 0:
                current_iter = iter + (epoch - 1) * self.train_iters_epoch
                tb_logger.add_scalar('train/navi_loss_train', navi_loss, current_iter)
                tb_logger.add_scalar('train/mi_loss_train', mi_loss, current_iter)
                tb_logger.add_scalar('train/ent_loss_train', ent_loss, current_iter)
                if traj is not None:
                    tb_logger.add_scalar('train/dist_from_goal', dist_from_goal, current_iter)
                    tb_logger.add_scalar('train/movements', movement, current_iter)

            if mode == "joint-reinforce":
                print('Epoch (RL stats): [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speaker reward {speaker_reward:.4f}\t'
                      'Navi reward {navi_reward:.4f}\t'
                      'Teacher forcing {use_teacher_forcing:.1f}\t'.format(
                    epoch, iter, self.train_iters_epoch, batch_time=batch_time, speaker_reward=speaker_reward,
                    navi_reward=navi_reward, use_teacher_forcing=float(use_teacher_forcing)))

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Navi Loss {navi_loss.val:.4f} ({navi_loss.avg:.4f})\t'
                  'MI Loss {mi_loss.val:.4f} ({mi_loss.avg:.4f})\t'.format(
                epoch, iter, self.train_iters_epoch, batch_time=batch_time, mi_loss=mi_losses,
                navi_loss=navi_losses)) #, nmt_loss=nmt_losses, mi_loss=mi_losses))

        if tb_logger:
            tb_logger.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            tb_logger.add_scalar('epoch/train/navi_loss', navi_losses.avg, epoch)
            tb_logger.add_scalar('epoch/train/mi_loss', mi_losses.avg, epoch)
            if generate_traj:
                tb_logger.add_scalar('epoch/train/dist_from_goal', dists.avg, epoch)
                tb_logger.add_scalar('epoch/train/movements', movements.avg, epoch)

    
    def eval(self, epoch, val_env, val_loader=None, use_val_loader=False, tb_logger=None, evaluation_mode=None):
        if evaluation_mode == 'self-eval-loader':
            assert(self.agent.speaker is not None)
            assert(use_val_loader == 1 and val_loader is not None)
        elif evaluation_mode == 'self-eval-R2R-env':
            assert(self.agent.speaker is not None)
            assert(use_val_loader == 0 and val_env is not None)
        elif evaluation_mode == 'trans-eval':
            assert(self.agent.translator is not None)
            assert(use_val_loader == 0 and val_env is not None)
        elif evaluation_mode == 'standard-eval-env':
            assert(use_val_loader == 0 and val_env is not None)
        elif evaluation_mode == 'joint-eval' or evaluation_mode == 'joint-speaker_DA-eval':
            assert(val_loader is not None or val_env is not None)

        batch_time = AverageMeter()
        navi_losses = AverageMeter()
        mi_losses   = AverageMeter()
        entropies   = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()

        val_navi_losses = AverageMeter()
        val_mi_losses = AverageMeter()
        val_acces = AverageMeter()

        if use_val_loader:
            env_name, (env, evaluator) = val_loader
        else:
            env_name, (env, evaluator) = val_env
        print('Evaluating on {} env'.format(env_name))

        self.agent.env = env
        '''
           we need to reset env everytime before start
           both seed and path_id
        '''
        if use_val_loader:
            self.agent.env.sampler.reset_path_id()
        else:
            self.agent.env.reset_epoch()

        if self.agent.translator is not None:
            self.agent.translator.eval()
        if self.agent.speaker is not None and not(self.opts.training_mode == 'joint-speaker_DA' or self.opts.training_mode == 'joint-speaker_DA-teacher' or self.opts.training_mode == 'joint-speaker_DA-teacher-student'):
            self.agent.speaker.eval()
        self.agent.encoder.eval()
        self.agent.model.eval()

        self.agent.feedback = self.opts.feedback
        self.agent.val_acc = None

        # load dataset path for computing ground truth distance
        if use_val_loader:
            val_iters_epoch = 100
            self.agent.gt = OrderedDict()
            self.agent.results = OrderedDict()
        else:
            self.agent.gt = OrderedDict()
            for item in load_datasets([env_name], dataset_name=self.opts.dataset_name):
                self.agent.gt[item['path_id']] = item
            val_iters_epoch = math.ceil(len(env.data) / self.opts.batch_size)
            self.agent.results = OrderedDict()
        
        looped = False
        iter = 1

        with torch.no_grad():
            end = time.time()
            while True:

                if self.opts.eval_beam:
                    raise ValueError('Not implemented; To be added')
                    # traj = self.agent.sample_beam(self.opts.beam_size)
                else:
                    # rollout the agent
                    if 'self-eval' in evaluation_mode:
                        mi_loss, traj = self.agent.rollout_speaker_cogrounding(use_loader=use_val_loader, train=False)
                        navi_loss = None
                    elif 'trans-eval' in evaluation_mode:
                        navi_loss, traj = self.agent.rollout_cogrounding(train=False)
                        mi_loss = None
                    elif evaluation_mode == 'standard-eval-env':
                        navi_loss, traj = self.agent.rollout_cogrounding(train=False)
                        mi_loss = None
                    elif evaluation_mode == 'joint-eval':
                        if use_val_loader:
                          mi_loss, traj = self.agent.rollout_speaker_cogrounding(use_loader=use_val_loader, train=False)
                          navi_loss = None
                        else:
                          navi_loss, traj = self.agent.rollout_cogrounding(train=False)
                          mi_loss = None
                    elif evaluation_mode == 'joint-speaker_DA-eval':
                        if use_val_loader:
                          mi_loss, traj = self.agent.rollout_SF_speaker_cogrounding(use_loader=use_val_loader, train=False)
                          navi_loss = None
                        else:
                          navi_loss, traj = self.agent.rollout_cogrounding(train=False)
                          mi_loss = None
                    else:
                        raise ValueError('Wrong mode')

                    dist_from_goal = np.mean(self.agent.dist_from_goal)
                    movement = np.mean(self.agent.traj_length)

                    if navi_loss is not None:
                        navi_losses.update(navi_loss.item(), self.opts.batch_size)
                    if mi_loss is not None:
                        mi_losses.update(mi_loss.item(), self.opts.batch_size)
                    dists.update(dist_from_goal, self.opts.batch_size)
                    movements.update(movement, self.opts.batch_size)
                    if self.agent.val_acc is not None:
                        val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

                    if tb_logger and iter % 10 == 0:
                        current_iter = iter + (epoch - 1) * val_iters_epoch
                        tb_logger.add_scalar('{}/navi_loss'.format(env_name), navi_loss, current_iter)
                        tb_logger.add_scalar('{}/mi_loss'.format(env_name), mi_loss, current_iter)
                        tb_logger.add_scalar('{}/dist_from_goal'.format(env_name), dist_from_goal, current_iter)
                        tb_logger.add_scalar('{}/movements'.format(env_name), movement, current_iter)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Navi Loss {navi_loss.val:.4f} ({navi_loss.avg:.4f})\t'
                      'MI Loss {mi_loss.val:.4f} ({mi_loss.avg:.4f})\t'.format(
                    epoch, iter, val_iters_epoch, batch_time=batch_time, mi_loss=mi_losses,
                    navi_loss=navi_losses))#, nmt_loss=nmt_losses, mi_loss=mi_losses))

                # write into results
                if use_val_loader and iter == val_iters_epoch: # If using dataloader, all instr_id are unique, so we need another exit condition
                    looped = True
                for traj_ in traj:
                    if traj_['instr_id'] in self.agent.results:
                        looped = True
                    else:
                        result = OrderedDict([
                            ('path', traj_['path']),
                            ('path_id', traj_['path_id']),
                            ('distance', traj_['distance']),
                            ('img_attn', traj_['img_attn']),
                            ('ctx_attn', traj_['ctx_attn']),
                            ('viewpoint_idx', traj_['viewpoint_idx']),
                            ('navigable_idx', traj_['navigable_idx'])
                        ])
                        self.agent.results[traj_['instr_id']] = result
                if looped:
                    break
                iter += 1

        if tb_logger:
            tb_logger.add_scalar('epoch/{}/navi_loss'.format(env_name), navi_losses.avg, epoch)
            tb_logger.add_scalar('epoch/{}/mi_loss'.format(env_name), mi_losses.avg, epoch)
            tb_logger.add_scalar('epoch/{}/dist_from_goal'.format(env_name), dists.avg, epoch)
            tb_logger.add_scalar('epoch/{}/movements'.format(env_name), movements.avg, epoch)
            if self.agent.val_acc is not None:
                tb_logger.add_scalar('epoch/{}/val_acc'.format(env_name), val_acces.avg, epoch)

        # dump into JSON file
        if self.opts.eval_beam:
            self.agent.results_path = '{}{}-beam_{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                             self.opts.beam_size, env_name, epoch)
        else:
            self.agent.results_path = '{}{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                     env_name, epoch)
        self.agent.write_results()
        if use_val_loader:
            score_summary, _ = evaluator.score_online(self.agent.results_path, self.agent.gt)
        else:
            score_summary, _ = evaluator.score(self.agent.results_path)
        result_str = ''
        success_rate = 0.0
        for metric, val in score_summary.items():
            result_str += '| {}: {} '.format(metric, val)
            if metric in ['success_rate']:
                success_rate = val
            if tb_logger:
                tb_logger.add_scalar('score/{}/{}'.format(env_name, metric), val, epoch)
        print(result_str)

        return success_rate
    
