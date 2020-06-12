import json
import random
import numpy as np
import copy

from collections import OrderedDict
import sys
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import padding_idx, end_token_idx, pad_list_tensors, kl_div, CrossEntropy
#from data_management import MyQueue, NodeState

import pdb
class PanoBaseAgent(object):
    """ Base class for an R2R agent with panoramic view and action. """

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = OrderedDict()
    
    def write_results(self):
        output = []
        for k, v in self.results.items():
            output.append(
                OrderedDict([
                    ('instr_id', k),
                    ('trajectory', v['path']),
                    ('distance', v['distance']),
                    ('img_attn', v['img_attn']),
                    ('ctx_attn', v['ctx_attn']),
                    ('viewpoint_idx', v['viewpoint_idx']),
                    ('navigable_idx', v['navigable_idx']),
                    ('path_id', v['path_id']),
                ])
            )
        with open(self.results_path, 'w') as f:
            json.dump(output, f)
    
    def _get_distance(self, ob):
        try:
            gt = self.gt[int(ob['instr_id'].split('_')[0])]
        except:  # synthetic data only has 1 instruction per path
            gt = self.gt[int(ob['instr_id'])]
        distance = self.env.distances[ob['scan']][ob['viewpoint']][gt['path'][-1]]
        return distance

    def _select_action(self, logit, ended, is_prob=False, fix_action_ended=True):
        logit_cpu = logit.clone().cpu()
        if is_prob:
            probs = logit_cpu
        else:
            probs = F.softmax(logit_cpu, 1)

        if self.feedback == 'argmax':
            _, action = probs.max(1)  # student forcing - argmax
            action = action.detach()
        elif self.feedback == 'sample':
            # sampling an action from model
            m = D.Categorical(probs)
            action = m.sample()
        else:
            raise ValueError('Invalid feedback option: {}'.format(self.feedback))

        # set action to 0 if already ended
        if fix_action_ended:
            for i, _ended in enumerate(ended):
                if _ended:
                    action[i] = self.ignore_index

        return action

    def _next_viewpoint(self, obs, viewpoints, navigable_index, action, ended):
        next_viewpoints, next_headings = [], []
        next_viewpoint_idx = []

        for i, ob in enumerate(obs):
            if action[i] >= 1 and not action[i] == self.ignore_index:
                next_viewpoint_idx.append(navigable_index[i][action[i] - 1])  # -1 because the first one in action is 'stop'
            else:
                next_viewpoint_idx.append('STAY')
                ended[i] = True

            # use the available viewpoints and action to select next viewpoint
            next_action = 0 if action[i] == self.ignore_index else action[i]
            next_viewpoints.append(viewpoints[i][next_action])
            # obtain the heading associated with next viewpoints
            next_headings.append(ob['navigableLocations'][next_viewpoints[i]]['heading'])

        return next_viewpoints, next_headings, next_viewpoint_idx, ended

    def pano_navigable_feat(self, obs, ended, next_path_idx=None):

        # Get the 36 image features for the panoramic view (including top, middle, bottom)
        num_feature, feature_size = obs[0]['feature'].shape

        pano_img_feat = torch.zeros(len(obs), num_feature, feature_size)
        navigable_feat = torch.zeros(len(obs), self.opts.max_navigable, feature_size)

        navigable_feat_index, target_index, viewpoints = [], [], []
        for i, ob in enumerate(obs):
            pano_img_feat[i, :] = torch.from_numpy(ob['feature'])  # pano feature: (batchsize, 36 directions, 2048)

            index_list = []
            viewpoints_tmp = []
            gt_viewpoint_id, viewpoint_idx = ob['gt_viewpoint_idx']
            # If we want to follow the GT traj exactly, we can't rely on the targets found above.
            if next_path_idx is not None:
                goal_viewpoint = ob['teacher'][next_path_idx[i]]
                # If we have reached the current objective but haven't reached the end of the path,
                # increment path_idx to set next step as objective.
                if ob['viewpoint'] == goal_viewpoint and next_path_idx[i] < len(ob['teacher'])-1:
                    next_path_idx[i] += 1
                    goal_viewpoint = ob['teacher'][next_path_idx[i]]
                teacher_path = self.env.paths[ob['scan']][ob['viewpoint']][goal_viewpoint]
                if len(teacher_path) > 1:
                    gt_viewpoint_id = teacher_path[1]
                else:
                    # Due to the previous if statement, this is only possible if the current viewpoint
                    # has reached the end of the entire path.
                    gt_viewpoint_id = ob['viewpoint']
            

            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                index_list.append(int(ob['navigableLocations'][viewpoint_id]['index']))
                viewpoints_tmp.append(viewpoint_id)

                if viewpoint_id == gt_viewpoint_id:
                    # if it's already ended, we label the target as <ignore>
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)

            # we ignore the first index because it's the viewpoint index of the current location
            # not the viewpoint index for one of the navigable directions
            # we will use 0-vector to represent the image feature that leads to "stay"
            navi_index = index_list[1:]
            navigable_feat_index.append(navi_index)
            viewpoints.append(viewpoints_tmp)

            navigable_feat[i, 1:len(navi_index) + 1] = pano_img_feat[i, navi_index]

        return pano_img_feat, navigable_feat, (viewpoints, navigable_feat_index, target_index)

    def teacher_forcing_target(self, step, obs, ended):
        target_index = []
        for i, ob in enumerate(obs):
            gt_viewpoint_id = ob['teacher'][min(step+1, len(ob['teacher'])-1)]
            for j, viewpoint_id in enumerate(ob['navigableLocations']):
                if viewpoint_id == gt_viewpoint_id:
                    if ended[i] and self.opts.use_ignore_index:
                        target_index.append(self.ignore_index)
                    else:
                        target_index.append(j)
        return target_index 

    def _sort_batch(self, obs, only_keep_five=False):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        if only_keep_five:
            num_keep = 5
            for i, row in enumerate(seq_tensor):
                seq_length = seq_lengths[i]
                if seq_length <= num_keep + 2: # One extra for start, and one for end       
                    continue
                if seq_length == seq_tensor.shape[1]: # Rare edge case, but need to add end token
                    seq_tensor[i,1:num_keep+1] = row[seq_length-num_keep:seq_length]
                    seq_tensor[i, num_keep+1]  = end_token_idx 
                else:
                    seq_tensor[i,1:7] = row[seq_length-num_keep-1:seq_length]
                seq_tensor[i, 7:] = padding_idx
                seq_lengths[i] = num_keep + 2
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)

    def _sort_batch_from_seq(self, seq_list):
        """ Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). """
        seq_tensor = np.array(seq_list)
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length
        seq_tensor = torch.from_numpy(seq_tensor)
        return seq_tensor.long().to(self.device), list(seq_lengths)

class PanoSeq2SeqAgent(PanoBaseAgent):
    """ An agent based on an LSTM seq2seq model with attention. """
    def __init__(self, opts, env, results_path, encoder, model, translator=None, backtranslator=None, feedback='sample', episode_len=20, speaker=None, monte_carlo_translator=False, monte_carlo_speaker=False, reward_strategy='dtw+sr', actor_network=None, q_networks=None):
        super(PanoSeq2SeqAgent, self).__init__(env, results_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.encoder = encoder
        self.model = model
        self.actor_network = actor_network
        self.q_networks = q_networks
        self.translator = translator
        self.backtranslator = backtranslator
        self.speaker = speaker
        self.feedback = feedback
        self.episode_len = episode_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monte_carlo_translator = monte_carlo_translator
        self.monte_carlo_speaker    = monte_carlo_speaker
        
        self.ignore_index = opts.max_navigable + 1  # we define (max_navigable+1) as ignore since 15(navigable) + 1(STOP)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        if self.opts.use_backtranslator:
            self.backtran_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        if self.opts.use_nmt_loss:
            self.nmt_criterion = nn.CrossEntropyLoss()
        self.cross_entropy = CrossEntropy()
        
        self.MSELoss = nn.MSELoss()
        self.MSELoss_sum = nn.MSELoss(reduction='sum')

        self.cur_iteration = 0

        # RL settings
        self.rewards = {}
        self.rav_step = [0] * self.episode_len
        self.rav_step_count = [0] * self.episode_len
        self.reward_strategy = reward_strategy

    def init_traj(self, obs):
        """initialize the trajectory"""
        batch_size = len(obs)

        traj, scan_id = [], []
        for ob in obs:
            traj.append(OrderedDict([
                ('instr_id', ob['instr_id']),
                ('path_id', ob['path_id'] if 'path_id' in ob.keys() else None),
                ('path', [(ob['viewpoint'], ob['heading'], ob['elevation'])]),
                ('length', 0),
                ('feature', [ob['feature']]),
                ('img_attn', []),
                ('ctx_attn', []),
                ('action_confidence', []),
                ('regret', []),
                ('viewpoint_idx', []),
                ('navigable_idx', []),
                ('gt_viewpoint_idx', ob['gt_viewpoint_idx']),
                ('steps_required', [len(ob['teacher'])]),
                ('distance', [super(PanoSeq2SeqAgent, self)._get_distance(ob)]),
                ('reward', []),
                ('acc_reward', []),
                ('reward_sr', []),
                ('reward_cls', []),
                ('reward_dtw', []),
                ('reward_intrinsic', []),
                ('reward_distance', []),
                ('logit', []),
                ('action', []),
                ('value', []),
            ]))
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        self.value_loss = torch.tensor(0).float().to(self.device)

        ended = np.array([False] * batch_size)
        last_recorded = np.array([False] * batch_size)

        self.rav_speaker_step = [0] * self.episode_len
        self.rav_speaker_step_count = [0] * self.episode_len
        self.rav_translator_step = [0] * self.episode_len
        self.rav_translator_step_count = [0] * self.episode_len

        return traj, scan_id, ended, last_recorded

    def update_traj(self, obs, traj, img_attn, ctx_attn, next_viewpoint_idx,
                    navigable_index, ended, last_recorded, action_prob=None,
                    logit=None, action=None, cur_step=-1, rav_type='speaker',
                    is_last_step=False, value=None):
        # Save trajectory output and rewards
        for i, ob in enumerate(obs):
            if not ended[i] or not last_recorded[i]:
                traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                traj[i]['distance'].append(dist)
                traj[i]['img_attn'].append(img_attn[i].detach().cpu().numpy().tolist())
                traj[i]['ctx_attn'].append(ctx_attn[i].detach().cpu().numpy().tolist())

                if action_prob is not None:
                    traj[i]['action_confidence'].append(action_prob[i].detach().cpu().item())
                if value is not None:
                    if len(value[1]) > 1:
                        traj[i]['value'].append(value[i].detach().cpu().tolist())
                    else:
                        traj[i]['value'].append(value[i].detach().cpu().item())

                traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                traj[i]['navigable_idx'].append(navigable_index[i])
                traj[i]['steps_required'].append(len(ob['new_teacher']))
                self.traj_length[i] = self.traj_length[i] + 1
                last_recorded[i] = True if ended[i] else False

                if last_recorded[i] or is_last_step:
                    traj[i]['reward_intrinsic'].append(0)
                    traj[i]['reward_distance'].append(traj[i]['distance'][-1] - traj[i]['distance'][-2])
                    generated_path = [ele[0] for ele in traj[i]['path']]
                    try:
                        reference_path = self.gt[traj[i]['path_id']]['path']
                    except:
                        reference_path = ob['teacher']
                    traj[i]['reward_sr'].append(self.env.get_sr(ob['scan'], generated_path, reference_path))
                    traj[i]['reward_cls'].append(self.env.get_cls(ob['scan'], generated_path, reference_path))
                    traj[i]['reward_dtw'].append(self.env.get_dtw(ob['scan'], generated_path, reference_path))
                else:
                    traj[i]['reward_intrinsic'].append(0)
                    traj[i]['reward_distance'].append(traj[i]['distance'][-1] - traj[i]['distance'][-2])
                    traj[i]['reward_sr'].append(0)
                    traj[i]['reward_cls'].append(0)
                    traj[i]['reward_dtw'].append(0)
                if self.reward_strategy == 'cls+sr':
                    reward = traj[i]['reward_cls'][-1] + traj[i]['reward_sr'][-1]
                elif self.reward_strategy == 'dtw+sr':
                    reward = traj[i]['reward_dtw'][-1] + traj[i]['reward_sr'][-1]
                elif self.reward_strategy == 'distance+sr':
                    if not last_recorded[i] and not is_last_step:
                        reward = traj[i]['reward_distance'][-1]
                    else:
                        reward = traj[i]['reward_sr'][-1]
                else:
                    assert(1==0)
                traj[i]['reward'].append(reward)
                if logit is not None:
                    traj[i]['logit'].append(logit[i:i+1]) # logit: [N, num_actions]
                if action is not None:
                    traj[i]['action'].append(action[i:i+1].detach()) # action: [N]

        return traj, last_recorded

    def compute_acc_reward(self, traj, gamma=0.95):
        # get advantages
        for i, _ in enumerate(traj):
            t_len = len(traj[i]['reward'])
            reward_all = np.array(traj[i]['reward']).copy()
            R_extr = reward_all[-1]
            for j in range(t_len - 1):
                R_extr = R_extr * gamma + reward_all[t_len - j - 2]
                reward_all[t_len - j - 2] = R_extr
            traj[i]['acc_reward'] = reward_all

        return traj

    def get_reward_loss(self, traj, rav_type, gamma=0.95):
        traj = self.compute_acc_reward(traj, gamma)
        reward_loss = 0
        reward = 0

        for i, traj_ in enumerate(traj):
            acc_reward = traj_['acc_reward']
            for cur_step in range(0, len(acc_reward)):
                if  rav_type == 'speaker':
                    self.rav_speaker_step[cur_step] = (self.rav_speaker_step[cur_step] * self.rav_speaker_step_count[cur_step] + acc_reward[cur_step]) / (
                                                       self.rav_speaker_step_count[cur_step] + 1)
                    self.rav_speaker_step_count[cur_step] += 1
                elif rav_type == 'translator':
                    self.rav_translator_step[cur_step] = (self.rav_translator_step[cur_step] * self.rav_translator_step_count[cur_step] + acc_reward[cur_step]) / (
                                                          self.rav_translator_step_count[cur_step] + 1)
                    self.rav_translator_step_count[cur_step] += 1
                else:
                    assert(1==0)

        for i, traj_ in enumerate(traj):
            acc_reward = traj_['acc_reward']
            adv = torch.Tensor(acc_reward).to(self.device).clone()
            if  rav_type == 'speaker':
                rav_list = self.rav_speaker_step
            elif rav_type == 'translator':
                rav_list = self.rav_translator_step
            else:
                assert(1==0)
            for cur_step in range(0, len(acc_reward)):
                adv[cur_step] = adv[cur_step] - rav_list[cur_step]
            action = torch.cat(traj_['action'], dim=0).to(self.device)
            logit = torch.cat(traj_['logit'], dim=0).to(self.device)
            ce_loss = F.cross_entropy(logit, action, reduction='none', ignore_index=self.ignore_index)

            reward_loss += (ce_loss * adv).sum()
            reward += torch.Tensor(traj_['reward']).to(self.device).sum()

        return reward_loss / len(traj), reward / len(traj)
            
    def update_cur_iteration(self, cur_iteration):
        self.cur_iteration = cur_iteration
     
    def machine_translate(self):
        assert(None not in [self.translator, self.speaker])
        self.speaker.update_tau(iteration=self.cur_iteration)
        self.translator.update_tau(iteration=self.cur_iteration)

        obs = np.array(self.env.reset())
        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        all_img_feats, all_action_embeddings = self.env.get_path_features(obs, return_action_embeddings=True)

        # Generate target neuralese, and logits of translator
        pseudo_seq, \
        _,          \
        _,          \
        _           = self.speaker(all_img_feats, all_action_embeddings)
        target      = pseudo_seq.detach().max(dim=-1)[1]
        _, _, _, logits = self.translator(seq, seq_lengths, provided_pred_seq=pseudo_seq)
        # Get teacher forcing Cross Entropy loss
        nmt_loss  = self.nmt_criterion(logits, target)
        return nmt_loss, None

    def rollout_speaker_cogrounding(self, use_loader=True, train=True, teacher_forcing=False):
    # Grab observations from loader, and produce neuralese descriptions
        if use_loader: 
            if train:
                self.env.sampler.reset_path_id() # so that we don't explode memory from sampling
            obs, items = self.env.reset()
            obs = np.array(obs)
            for item in items:
                self.gt[item['path_id']] = item
        else:
            obs = np.array(self.env.reset())
        batch_size         = len(obs)
        self.speaker.update_tau(iteration=self.cur_iteration)
        all_img_feats, all_action_embeddings = self.env.get_path_features(obs, return_action_embeddings=True)
        #s_enc_inputs = {'all_img_feats': all_img_feats, 'all_action_embeddings': all_action_embeddings}
        pseudo_seq,         \
        pseudo_seq_lengths, \
        extra_loss,         \
        logits              = self.speaker(all_img_feats, all_action_embeddings)
        if self.cur_iteration % 200 == 0:
        #  print(pseudo_seq[0:3,0:8])
          print('pseudo seq len:', pseudo_seq_lengths[0:10])
          print('pseudo seq ex:', pseudo_seq[0,...].max(dim=-1)[1])
        #  print('Entropy:', - extra_loss)
            
        # Encode neuralese and set up for decoding using CoGrounding
        ctx,           \
        h_t,           \
        c_t,           \
        ctx_mask       = self.encoder(pseudo_seq, pseudo_seq_lengths)
        question       = h_t
        pre_feat       = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)
        traj,          \
        scan_id,       \
        ended,         \
        last_recorded  = self.init_traj(obs)
        mi_loss        = 0 + extra_loss
        # Decoding step
        if self.opts.follow_gt_traj:
            next_path_idx = np.ones(batch_size, dtype=np.uint8)
        else:
            next_path_idx = None
        for step in range(self.opts.max_episode_len):
            # Get environment information at current step
            pano_img_feat,    \
            navigable_feat,   \
            viewpoint_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended, next_path_idx=next_path_idx)
            viewpoints,       \
            navigable_index,  \
            target_index      = viewpoint_indices
            # Follow sampled path rather than take closest path to goal
            if train and (self.opts.use_teacher_forcing or teacher_forcing):
                target_index  = super(PanoSeq2SeqAgent, self).teacher_forcing_target(step, obs, ended)
            pano_img_feat     = pano_img_feat.to(self.device)
            navigable_feat    = navigable_feat.to(self.device)
            target          = torch.LongTensor(target_index).to(self.device)

            # Calculate action logit.
            h_t,            \
            c_t,            \
            pre_ctx_attend, \
            img_attn,       \
            ctx_attn,       \
            logit,          \
            navigable_mask, \
            ve_loss         = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx, 
                pre_ctx_attend, navigable_index, ctx_mask)
            mi_loss += ve_loss * self.opts.ve_beta
            # Calculate loss, and move to next observation through teacher forcing.
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            mi_loss            += self.criterion(logit, target)
            if train and (self.opts.use_teacher_forcing or teacher_forcing):
                action = target.clone().detach() # teacher forcing
                if self.opts.use_ignore_index:
                    action[action == self.ignore_index] = 0 # If we've already reached the end, stay where we are.
            else:
                action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, fix_action_ended=self.opts.fix_action_ended)
            next_viewpoints,    \
            next_headings,      \
            next_viewpoint_idx, \
            ended               = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)
            obs           = self.env.step(scan_id, next_viewpoints, next_headings)
            traj,         \
            last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, next_viewpoint_idx,
                navigable_index, ended, last_recorded, logit=logit, action=action, cur_step=step,
                rav_type='speaker', is_last_step=(step==self.opts.max_episode_len-1))
            pre_feat      = navigable_feat[torch.LongTensor(range(batch_size)), action, :]

            if last_recorded.all():
                break

            self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]
        return mi_loss, traj

    def get_seq_list(self, DA_instr, tokenizer):
        seq_list = []
        for _, DA_ele in enumerate(DA_instr):
            raw_words = DA_ele['words']
            words = []
            for w in raw_words:
                if w != '<UNK>':
                    words += [w]
            sentence = (' ').join(words)
            seq_list.append(tokenizer.encode_sentence(sentence))
        return seq_list

    # rollout trajectories for SF speaker based data generation
    def rollout_SF_speaker_cogrounding(self, use_loader=True, train=True, teacher_forcing=False):
        # Grab observations from loader, and produce neuralese descriptions
        if use_loader: 
            if train:
                self.env.sampler.reset_path_id() # so that we don't explode memory from sampling
            obs, items = self.env.reset()
            obs = np.array(obs)
            for item in items:
                self.gt[item['path_id']] = item
        else:
            assert(1==0)
        batch_size         = len(obs)
        all_img_feats, all_action_embeddings = self.env.get_path_features(obs, return_action_embeddings=True)

        DA_instr = self.speaker.test(use_dropout=False, feedback='argmax', \
                                     info_list=[obs, all_img_feats, all_action_embeddings])
        seq_list = self.get_seq_list(DA_instr, self.env.loader_tokenizer)
        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch_from_seq(seq_list)

        if self.cur_iteration % 50 == 0:
            for i in range(3):
                sentence = (' ').join(DA_instr[i]['words'])
                print('sentence ', i, sentence)
            
        # Encode neuralese and set up for decoding using CoGrounding
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        question       = h_t
        pre_feat       = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)
        traj, scan_id, ended, last_recorded  = self.init_traj(obs)

        mi_loss        = 0
        # Decoding step
        for step in range(self.opts.max_episode_len):

            # Get environment information at current step
            pano_img_feat, navigable_feat,   \
            viewpoint_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoint_indices
            # Follow sampled path rather than take closest path to goal
            if train and (self.opts.use_teacher_forcing or teacher_forcing):
                target_index  = super(PanoSeq2SeqAgent, self).teacher_forcing_target(step, obs, ended)
            pano_img_feat     = pano_img_feat.to(self.device)
            navigable_feat    = navigable_feat.to(self.device)
            target          = torch.LongTensor(target_index).to(self.device)

            # Calculate action logit.
            h_t,            \
            c_t,            \
            pre_ctx_attend, \
            img_attn,       \
            ctx_attn,       \
            logit,          \
            navigable_mask, \
            ve_loss         = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx, 
                pre_ctx_attend, navigable_index, ctx_mask)
            if self.opts.env_drop_follower:
                question = pre_ctx_attend
            mi_loss += ve_loss * self.opts.ve_beta
            
            # Calculate loss, and move to next observation through teacher forcing.
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            mi_loss            += self.criterion(logit, target)
            if train and (self.opts.use_teacher_forcing or teacher_forcing):
                action = target.clone().detach() # teacher forcing
                if self.opts.use_ignore_index:
                    action[action == self.ignore_index] = 0 # If we've already reached the end, stay where we are.
            else:
                action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, fix_action_ended=self.opts.fix_action_ended)

            next_viewpoints,    \
            next_headings,      \
            next_viewpoint_idx, \
            ended               = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                                            obs, viewpoints, navigable_index, action, ended)
            obs           = self.env.step(scan_id, next_viewpoints, next_headings)

            traj,         \
            last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, next_viewpoint_idx,
                navigable_index, ended, last_recorded, logit=logit, action=action, cur_step=step,
                rav_type='speaker', is_last_step=(step==self.opts.max_episode_len-1))

            pre_feat      = navigable_feat[torch.LongTensor(range(batch_size)), action, :]

            if last_recorded.all():
                break

            self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return mi_loss, traj


    def rollout_cogrounding(self, use_loader=False, vae_like=False, train=True, teacher_forcing=False):
        if use_loader:
            if train:
                try:
                    self.env.sampler.reset_path_id() # so that we don't explode memory from sampling
                except:
                    pass
            obs, items = self.env.reset(return_batch=True)
            obs = np.array(obs)
            for item in items:
                self.gt[item['path_id']] = item
        else:
            obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs, only_keep_five=self.opts.only_keep_five)
        if self.translator is not None:
          self.translator.update_tau(iteration=self.cur_iteration)
          pseudo_seq, pseudo_seq_lengths, extra_loss, logit_p_gen = self.translator(seq, seq_lengths, train=train)
          
          if self.cur_iteration % 50 == 0:
            print('pseudo seq', pseudo_seq[0:3,0:8])
            print('pseudo seq len:', pseudo_seq_lengths[0:10])
            p_gen_print = F.softmax(logit_p_gen, dim=1).transpose(1,2)
            torch.set_printoptions(profile="full")
            print('pseudo seq probability:', p_gen_print[0:3,0:8])
            torch.set_printoptions(profile="default")
            print('seq len:', seq_lengths[0:10])
            print('Entropy:', - extra_loss)
          '''
          if self.backtranslator is not None:
            seq_one_hot = torch.eye(self.backtranslator.decoder.vocab_size)[seq].to(self.device).detach()
            self.backtranslator.update_tau(iteration=self.cur_iteration)
            #bt_enc_inputs = {'inputs':pseudo_seq, 'lengths': pseudo_seq_lengths}
            #logits = self.backtranslator(bt_enc_inputs, provided_pred_seq=seq_one_hot)
            _, _, backtran_ent_loss, logit_back = self.backtranslator(pseudo_seq, pseudo_seq_lengths, provided_pred_seq=seq_one_hot)
            extra_loss += self.backtran_criterion(logit_back, seq) * self.opts.backtranslator_beta + backtran_ent_loss
            if self.cur_iteration % 200 == 0:
              print('seq:', seq[0,:])
              print('pseudo seq:', pseudo_seq[0,...].max(dim=-1)[1])
              print('greedy backtranslated seq:', logit_back[0,...].max(dim=0)[1])
            if self.opts.cyclic_reconstruction_only:
              return extra_loss, None
          '''
          ctx, h_t, c_t, ctx_mask = self.encoder(pseudo_seq, pseudo_seq_lengths)
          if self.speaker is not None: # Add in JSD regularization
            self.speaker.update_tau(iteration=self.cur_iteration)
            all_img_feats, all_action_embeddings = self.env.get_path_features(obs, return_action_embeddings=True)
            s_pseudo_seq, s_pseudo_seq_lengths, _, logit_q_gen = self.speaker(all_img_feats, all_action_embeddings)
            if self.cur_iteration % 50 == 0:
              print('speaker pseudo seq', s_pseudo_seq[0:3,0:8])
              s_q_gen_print = F.softmax(logit_q_gen, dim=1).transpose(1,2)
              torch.set_printoptions(profile="full")
              print('speaker pseudo seq probability:', s_q_gen_print[0:3,0:8])
              torch.set_printoptions(profile="default")
            # back-translate from speaker to natural language
            if self.backtranslator is not None:
              seq_one_hot = torch.eye(self.backtranslator.decoder.vocab_size)[seq].to(self.device).detach()
              self.backtranslator.update_tau(iteration=self.cur_iteration)
              _, _, _, logit_back = self.backtranslator(s_pseudo_seq, s_pseudo_seq_lengths, provided_pred_seq=seq_one_hot)
              extra_loss += self.backtran_criterion(logit_back, seq) * self.opts.backtranslator_beta
              if self.cur_iteration % 50 == 0:
                print('seq:', seq[0,:])
                print('pseudo seq:', pseudo_seq[0,...].max(dim=-1)[1])
                print('greedy backtranslated seq:', logit_back[0,...].max(dim=0)[1])
              if self.opts.cyclic_reconstruction_only:
                return extra_loss, None

            # Get probabilities of messages
            if self.opts.neuralese_student_forcing:
              _, _, _, logit_p_given = self.translator(seq, seq_lengths)
              _, _, _, logit_q_given = self.speaker(all_img_feats, all_action_embeddings)
            else:
              _, _, _, logit_p_given = self.translator(seq, seq_lengths, provided_pred_seq=s_pseudo_seq)
              _, _, _, logit_q_given = self.speaker(all_img_feats, all_action_embeddings, provided_pred_seq=pseudo_seq)
            # For compositional speaker:    hard to match translator teacher
            # For compositional translator: hard to match speaker teacher
            if self.opts.gumbel_hard:
                if self.monte_carlo_translator:
                  CE_term_0 = self.cross_entropy(pseudo_seq, logit_q_given, probability=True, seq_lengths=pseudo_seq_lengths)
                else:
                  CE_term_0 = self.cross_entropy(logit_p_gen, logit_q_given, seq_lengths=pseudo_seq_lengths)
                if self.monte_carlo_speaker:
                  CE_term_1 = self.cross_entropy(s_pseudo_seq, logit_p_given, probability=True, seq_lengths=s_pseudo_seq_lengths)
                else:
                  CE_term_1 = self.cross_entropy(logit_q_gen, logit_p_given, seq_lengths=s_pseudo_seq_lengths)
            else:
                CE_term_0 = ((s_pseudo_seq - pseudo_seq) * (s_pseudo_seq - pseudo_seq)).sum() / s_pseudo_seq.shape[0]
                CE_term_1 = 0
            CE_terms   = (CE_term_0, CE_term_1)
            if self.cur_iteration % 50 == 0:
              print('pq and qp losses: ', CE_term_0, CE_term_1)
            #print('P_gen: {}'.format(p_gen))
            #print('Q_gen: {}'.format(q_gen))
            #print('P_given: {}'.format(p_given))
            #print('Q_given: {}'.format(q_given))
            extra_loss += self.opts.kl_pq_beta * CE_terms[0] + self.opts.kl_qp_beta * CE_terms[1]
            if vae_like:
              return extra_loss
        else:
          ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
          extra_loss = 0
          
        if self.opts.replace_ctx_w_goal:
          all_img_feats, _ = self.env.get_path_features(obs, return_action_embeddings=True)
          img_feats_tensor = np.zeros((batch_size, 36, self.opts.img_feat_input_dim), np.float32)
          for i, img_feat in enumerate(all_img_feats):
            img_feats_tensor[i] = img_feat[-1]
          ctx = torch.from_numpy(img_feats_tensor).to(self.device)
          h_t = torch.zeros_like(h_t)
          c_t = torch.zeros_like(c_t)
          ctx_mask = torch.ones(batch_size, 36).to(self.device)

        question = h_t

        pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

        # Mean-Pooling over segments as previously attended ctx
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        loss = 0 + extra_loss
        if self.opts.follow_gt_traj:
            next_path_idx = np.ones(batch_size, dtype=np.uint8)
        else:
            next_path_idx = None
        for step in range(self.opts.max_episode_len):
            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended, next_path_idx=next_path_idx)
            viewpoints, navigable_index, target_index = viewpoints_indices

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)
            # Follow sampled path rather than take closest path to goal
            if train and (self.opts.use_teacher_forcing or teacher_forcing):
                target_index  = super(PanoSeq2SeqAgent, self).teacher_forcing_target(step, obs, ended)
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, navigable_mask, ve_loss = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                pre_ctx_attend, navigable_index, ctx_mask)
            if self.opts.env_drop_follower:
                question = pre_ctx_attend
            loss += ve_loss * self.opts.ve_beta

            # set other values to -inf so that logsoftmax will not affect the final computed loss
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)
            if train and (self.opts.use_teacher_forcing or teacher_forcing):
                action = target.clone().detach() # teacher forcing
                if self.opts.use_ignore_index:
                    action[action == self.ignore_index] = 0 # If we've already reached the end, stay where we are.
            else:
                action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                current_loss = current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded,
                                                   logit=logit, action=action, cur_step=step, 
                                                   rav_type='translator', is_last_step=(step==self.opts.max_episode_len))

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def rollout(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)
        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)

        pre_feat = torch.zeros(batch_size, obs[0]['feature'].shape[1]).to(self.device)

        # initialize the trajectory
        traj, scan_id = [], []
        for ob in obs:
            traj.append(OrderedDict([
                ('instr_id', ob['instr_id']),
                ('path', [(ob['viewpoint'], ob['heading'], ob['elevation'])]),
                ('length', 0),
                ('feature', [ob['feature']]),
                ('ctx_attn', []),
                ('viewpoint_idx', []),
                ('navigable_idx', []),
                ('gt_viewpoint_idx', ob['gt_viewpoint_idx']),
                ('steps_required', [len(ob['teacher'])]),
                ('distance', [super(PanoSeq2SeqAgent, self)._get_distance(ob)])
            ]))
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        ended = np.array([False] * len(obs))
        last_recorded = np.array([False] * len(obs))
        loss = 0
        if self.opts.follow_gt_traj:
            next_path_idx = np.ones(batch_size, dtype=np.uint8)
        else:
            next_path_idx = None
        for step in range(self.opts.max_episode_len):
            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended, next_path_idx)
            viewpoints, navigable_index, target_index = viewpoints_indices
            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)

            # get target
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, ctx_attn, logit, navigable_mask = self.model(pano_img_feat, navigable_feat, pre_feat, h_t, c_t, ctx, navigable_index, ctx_mask)

            # we mask out output
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

            loss += self.criterion(logit, target)

            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended)
            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # Save trajectory output and update last_recorded
            for i, ob in enumerate(obs):
                if not ended[i] or not last_recorded[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    dist = super(PanoSeq2SeqAgent, self)._get_distance(ob)
                    traj[i]['distance'].append(dist)
                    traj[i]['ctx_attn'].append(ctx_attn[i].detach().cpu().numpy().tolist())
                    traj[i]['viewpoint_idx'].append(next_viewpoint_idx[i])
                    traj[i]['navigable_idx'].append(navigable_index[i])
                    traj[i]['steps_required'].append(len(ob['new_teacher']))
                    self.traj_length[i] = self.traj_length[i] + 1
                    last_recorded[i] = True if ended[i] else False

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def rollout_monitor(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        question = h_t

        pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

        # Mean-Pooling over segments as previously attended ctx
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        loss = 0
        for step in range(self.opts.max_episode_len):

            pano_img_feat, navigable_feat, \
            viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)
            viewpoints, navigable_index, target_index = viewpoints_indices

            pano_img_feat = pano_img_feat.to(self.device)
            navigable_feat = navigable_feat.to(self.device)
            target = torch.LongTensor(target_index).to(self.device)

            # forward pass the network
            h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask, ve_loss = self.model(
                pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
                pre_ctx_attend, navigable_index, ctx_mask)

            loss += ve_loss * self.opts.ve_beta
            # set other values to -inf so that logsoftmax will not affect the final computed loss
            logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))
            current_logit_loss = self.criterion(logit, target)
            # select action based on prediction
            action = super(PanoSeq2SeqAgent, self)._select_action(logit, ended, fix_action_ended=self.opts.fix_action_ended)

            if not self.opts.test_submission:
                if step == 0:
                    current_loss = current_logit_loss
                else:
                    current_val_loss = self.get_value_loss_from_start(traj, value, ended)

                    self.value_loss += current_val_loss
                    current_loss = self.opts.value_loss_weight * current_val_loss + (
                            1 - self.opts.value_loss_weight) * current_logit_loss
            else:
                current_loss = torch.zeros(1)  # during testing where we do not have ground-truth, loss is simply 0
            loss += current_loss

            next_viewpoints, next_headings, next_viewpoint_idx, ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(
                obs, viewpoints, navigable_index, action, ended)

            # make a viewpoint change in the env
            obs = self.env.step(scan_id, next_viewpoints, next_headings)

            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded, value=value)

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj

    def get_value_loss_from_start(self, traj, predicted_value, ended, norm_value=True, threshold=5):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            value_target.append(dist_improved_from_start)

            if dist <= 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            # we will average the loss according to number of not 'ended', and use reduction='sum' for MSELoss
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        return self.MSELoss(predicted_value.squeeze(), value_target)

    def path_element_from_observation(ob):
        return (ob['viewpoint'], ob['heading'], ob['elevation'])

    def realistic_jumping(graph, start_step, dest_obs):
        if start_step == path_element_from_observation(dest_obs):
            return []
        s = start_step[0]
        t = dest_obs['viewpoint']
        path = nx.shortest_path(graph,s,t)
        traj = [(vp,0,0) for vp in path[:-1]]
        traj.append(path_element_from_observation(dest_obs))

        return traj

    def _GSA_env_step(self, obs, navigable_index, action, ended, scan_id):
        obs, viewpoints = world_states

        # Move to next point
        next_viewpoints, \
        next_headings, \
        next_viewpoint_idx, \
        ended = super(PanoSeq2SeqAgent, self)._next_viewpoint(obs, viewpoints, navigable_index, action, ended)

        # Step in env
        obs   = self.env.step(scan_id, next_viewpoints, next_headings)

        return obs, ended

    def _GSA_env_observe(self, obs, ended):
        pano_img_feat, navigable_feat, \
        viewpoints_indices = super(PanoSeq2SeqAgent, self).pano_navigable_feat(obs, ended)

        viewpoints, navigable_index, target_index = viewpoints_indices

        pano_img_feat = pano_img_feat.to(self.device)
        navigable_feat = navigable_feat.to(self.device)
        target = torch.LongTensor(target_index).to(self.device)

        env_features = (pano_img_feat, navigable_feat, navigable_index)
        world_states = (obs, viewpoints)

        return env_features, world_states

    def _GSA_model_step(self, ctx_features, env_features, model_states, pre_feat):
        # unpack ctx_features
        question, ctx, ctx_mask = ctx_features
        # unpack env states
        pano_img_feat, navigable_feat, navigable_index = env_features
        # unpack model states
        pre_ctx_attend, h_t, c_t = model_states

        # forward pass the network
        h_t, c_t, pre_ctx_attend, img_attn, ctx_attn, logit, value, navigable_mask = self.model(
            pano_img_feat, navigable_feat, pre_feat, question, h_t, c_t, ctx,
            pre_ctx_attend, navigable_index, ctx_mask)

        logit.data.masked_fill_((navigable_mask == 0).data, -float('inf'))

        next_model_states  = (pre_ctx_attend, h_t, c_t)
        next_action_states = (logit, navigable_mask)

        return next_model_states, next_action_states

    def _GSA_queue_push(self, world_states, model_states, action_states, env_features, \
                        ended, batch_queue, top_K, father_idx, pre_acc_logits):
        obs, view_points                       = world_states
        pre_ctx_attend, h_t, c_t               = model_states
        logit, navigable_mask                  = action_states
        pano_img_feat, navigable_feat, navigable_index = env_features

        batch_size = h_t.shape[0]
        for n in range(batch_size):
            logit_n      = logit[n]
            ranked_idx_n = logit_n.argsort(descending=True)
            pre_acc_logits_n = 0 if pre_acc_logits is None else pre_acc_logits[n]
            father_idx_n = -1 if father_idx is None else father_idx[n]
            for k in range(min(top_K, len(navigable_index[n]))):
                action_idx_k   = ranked_idx_n[k]
                action_emb_n_k = navigable_feat[n, action_idx_k]
                batch_queue[n].add(NodeState({'WorldState':  [obs[n], view_points[n], navigable_index[n]], \
                                              'ModelState':  [pre_ctx_attend[n], h_t[n], c_t[n]], \
                                              'ActionState': [logit_n[action_idx_k], action_emb_n_k, action_idx_k], \
                                              'FatherIdx':   father_idx_n, \
                                              'Value':       pre_acc_logits_n + logit_n[action_idx_k], \
                                              'EndedState':   ended[n],
                                             })
                                  )

        return batch_queue

    def _zip_reverse(self, in_list):
        out_list = [[] for _ in range(len(in_list[0]))]
        for i in range(len(in_list)):
            item = in_list[i]
            for j in range(len(item)):
                out_list[j].append(item[j])
        return out_list

    def _GSA_queue_batch(self, batch_queue, acts):
        world_states  = []
        model_states  = []
        action_states = []
        acc_logits    = []
        father_idx    = []
        ended         = []
        for i in range(len(batch_queue)):
            world_states  += [batch_queue[i].get(acts[i], 'WorldState')[0]]
            model_states  += [batch_queue[i].get(acts[i], 'ModelState')]
            action_states += [batch_queue[i].get(acts[i], 'ActionState')]
            acc_logits    += [batch_queue[i].get(acts[i], 'Value')]
            father_idx    += [batch_queue[i].get(acts[i], 'FatherIdx')]
            ended         += [batch_queue[i].get(acts[i], 'EndedState')]

        model_states  = self._zip_reverse(model_states)
        action_states = self._zip_reverse(action_states)

        return world_states, model_states, action_states, acc_logits, father_idx, ended

    def rollout_GSA(self):
        obs = np.array(self.env.reset())  # load a mini-batch
        batch_size = len(obs)

        seq, seq_lengths = super(PanoSeq2SeqAgent, self)._sort_batch(obs)

        ctx, h_t, c_t, ctx_mask = self.encoder(seq, seq_lengths)
        question = h_t

        pre_feat = torch.zeros(batch_size, self.opts.img_feat_input_dim).to(self.device)

        # Mean-Pooling over segments as previously attended ctx
        pre_ctx_attend = torch.zeros(batch_size, self.opts.rnn_hidden_size).to(self.device)

        # initialize the trajectory
        traj, scan_id, ended, last_recorded = self.init_traj(obs)

        # Initialize Queue
        static_ctx_features = (question, ctx, ctx_mask)
        batch_queue  = [MyQueue() for _ in range(batch_size)]
        env_features, world_states  = self._GSA_env_observe(obs, ended)
        model_states, action_states = self._GSA_model_step(ctx_features=static_ctx_features, \
                                                           env_features=env_features, \
                                                           model_states=(pre_ctx_attend, h_t, c_t), \
                                                           pre_feat=pre_feat)
        batch_queue  = self._GSA_queue_push(world_states, \
                                            model_states, \
                                            action_states, \
                                            env_features, \
                                            ended, \
                                            batch_queue, self.opts.GSA_top_K, father_idx=None, pre_acc_logits=None)

        # Rollout with Graph Search
        loss  = 0
        for step in range(self.opts.max_episode_len):
            # Get partial trajectories based on actions of Actor
            batch_queue_values = [batch_ele.get_by_key('Value') for batch_ele in batch_queue]
            # Remember to not repeat the previous traversed actions/partial trajs
            _, batch_actions = self.actor_network(batch_queue_values)

            world_states, model_states, action_states, acc_logits, father_idx, ended = self._GSA_queue_batch(batch_queue, batch_actions)


            # Add the teleport traj segments to trajs
            for i,ob in enumerate(last_obs):
                if not ended[i]:
                    last_vp = traj[i]['trajectory'][-1]
                    traj[i]['trajectory'] += realistic_jumping(
                        visit_graphs[i], last_vp, ob)

            # Expand on new nodes
            obs, ended = self._GSA_model_step(self, ctx_features, env_features, model_states, prev_action_emb)
            env_features, world_states  = self._GSA_env_observe(obs, ended)
            model_states, action_states = self._GSA_model_step(ctx_features=static_ctx_features, \
                                                               env_features=env_features, \
                                                               model_states=model_states, \
                                                               pre_feat=pre_feat)
            batch_queue  = self._GSA_queue_push(world_states, \
                                                model_states, \
                                                action_states, \
                                                batch_queue, \
                                                self.top_K, father_idx=None, prev_logits=None)


            # save trajectory output and update last_recorded
            traj, last_recorded = self.update_traj(obs, traj, img_attn, ctx_attn, next_viewpoint_idx,
                                                   navigable_index, ended, last_recorded, value=value)

            pre_feat = navigable_feat[torch.LongTensor(range(batch_size)), action,:]

            # Early exit if all ended
            if last_recorded.all():
                break

        self.dist_from_goal = [traj_tmp['distance'][-1] for traj_tmp in traj]

        return loss, traj
