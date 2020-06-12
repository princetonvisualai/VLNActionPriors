import torch
import torch.nn as nn
from collections import OrderedDict

from models.modules import build_mlp 


class GoalReconstructor(nn.Module):
    def __init__(self, opts, img_feat_input_dim, img_fc_dim, img_fc_use_batchnorm, img_dropout, fc_bias=True):
        super(GoalReconstructor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc_kwargs = OrderedDict([
            ('input_dim', img_feat_input_dim),
            ('hidden_dims', img_fc_dim),
            ('use_batchnorm', img_fc_use_batchnorm),
            ('dropout', img_dropout),
            ('fc_bias', fc_bias),
            ('relu', opts.mlp_relu)
        ]) 
        dec_kwargs = OrderedDict([
            ('input_dim', img_fc_dim[-1]),
            ('hidden_dims', img_fc_dim[:-1][::-1]+[img_feat_input_dim]),
            ('use_batchnorm', img_fc_use_batchnorm),
            ('dropout', img_dropout),
            ('fc_bias', fc_bias),
            ('relu', opts.mlp_relu)
        ]) 
        self.enc = build_mlp(**enc_kwargs)
        self.dec = build_mlp(**dec_kwargs)

        def forward(self, img_feat):
            encoding = self.enc(img_feat)
            pred     = self.dec(img_feat)
            return encoding, pred 

class GoalPredictor(nn.Module):
    def __init__(self, opts, img_feat_input_dim, ctx_input_dim, mlp_fc_dim, mlp_fc_use_batchnorm, mlp_dropout, fc_bias=True):
        super(GoalReconstructor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_kwargs = OrderedDict([
            ('input_dim', img_feat_input_dim + ctx_input_dim),
            ('hidden_dims', mlp_fc_dim),
            ('use_batchnorm', mlp_fc_use_batchnorm),
            ('dropout', mlp_dropout),
            ('fc_bias', fc_bias),
            ('relu', opts.mlp_relu)
        ]) 
        self.mlp = build_model(**mlp_kwargs)
        
    def forward(self, img_feat, ctx_feat):
        x = torch.cat([img_feat, ctx_feat], dim=1)
        pred = self.mlp(x)
        return pred
