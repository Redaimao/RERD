"""
author: Xiaobao
date: 2021-3-1
functionality: multimodal sequence fusion models
Model names: RERD
"""

import torch
from torch import nn
import torch.nn.functional as F
from modules.distillation_net import DistNet
from modules.transformer import TransformerEncoder
from src.sinkhorn import SinkhornDistance

from modules.BertTextEncoder import BertTextEncoder
import numpy as np


class RERDModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a RERD model.
        """
        super(RERDModel, self).__init__()

        # dimension/ projection params
        self.use_bert = hyp_params.use_bert
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
        # pre projection feature dim: predefined projection dims before using informer tokenembedding dim
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        # mark the input of each modality e.g., with lonly and vonly the inputs are language and vision
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly

        # encoder params
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        # distillation encoder params
        self.dis_factor = hyp_params.dis_factor
        self.dis_d_model = hyp_params.dis_d_model
        self.dis_n_heads = hyp_params.dis_n_heads
        self.dis_e_layers = hyp_params.dis_e_layers
        self.dis_d_ff = hyp_params.dis_d_ff
        self.dis_dropout = hyp_params.dis_dropout
        self.dis_attn = hyp_params.dis_attn

        # combine modality params
        self.partial_mode = self.lonly + self.aonly + self.vonly
        combined_dim, combined_type = self.get_combined_dim_and_type_with_pre_proj()

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.dis_d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.dis_d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.dis_d_model, kernel_size=1, padding=0, bias=False)
        # bert feature projection layers
        if self.use_bert:
            self.bert_feature_projection = nn.Sequential(
                nn.Conv1d(self.orig_d_l, self.d_l * 3, kernel_size=1, padding=0, bias=False),
                nn.Conv1d(self.d_l * 3, self.d_l, kernel_size=1, padding=0, bias=False)
            )

        self.l_proj_ori_dim, self.a_proj_ori_dim, self.v_proj_ori_dim, self.fusion_distill_len = self.get_distill_len()
        # self.temp_proj_dim = hyp_params.temp_proj
        self.temp_proj_dim = min([self.l_proj_ori_dim, self.a_proj_ori_dim, self.v_proj_ori_dim])

        # 1*: temporal projection layer: project temporal dim to the same dim:
        self.temp_proj_l = nn.Conv1d(self.l_proj_ori_dim, self.temp_proj_dim, kernel_size=1, padding=0, bias=False)
        self.temp_proj_a = nn.Conv1d(self.a_proj_ori_dim, self.temp_proj_dim, kernel_size=1, padding=0, bias=False)
        self.temp_proj_v = nn.Conv1d(self.v_proj_ori_dim, self.temp_proj_dim, kernel_size=1, padding=0, bias=False)

        # 3. and self quality improvement layers
        if 'l' in list(combined_type):
            self.l_refine = self.get_network(self_type='l_refine')
        if 'a' in list(combined_type):
            self.a_refine = self.get_network(self_type='a_refine')
        if 'v' in list(combined_type):
            self.v_refine = self.get_network(self_type='v_refine')

        # 4. Cross-modal Fusion
        # trimodal inputs
        self.cross_modal_fusion = self.get_fusion_network(self_type='cross_fusion')

        # 5. Projection layers: for classification or regression
        self.proj1 = nn.Linear(self.temp_proj_dim * combined_dim, self.temp_proj_dim * combined_dim)
        self.proj2 = nn.Linear(self.temp_proj_dim * combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # 6 sinkhorn regularization for multimodal embeddings
        self.sink_horn = SinkhornDistance(eps=0.5, max_iter=30)  # 0.1 100

        # pretrained bert:
        self.bert_model = BertTextEncoder(language='en', pretrain_weight=hyp_params.bert_path, use_finetune=False)

    def get_distill_len(self):
        l_dis_dim, a_dis_dim, v_dis_dim = self.orig_l_len, self.orig_a_len, self.orig_v_len
        for i in range(0, self.dis_e_layers - 1):
            l_dis_dim = np.floor((l_dis_dim + 1) / 2 + 1)
            a_dis_dim = np.floor((a_dis_dim + 1) / 2 + 1)
            v_dis_dim = np.floor((v_dis_dim + 1) / 2 + 1)
            # print('-----dis len---')
            # print(l_dis_dim, a_dis_dim, v_dis_dim)
        return np.int(l_dis_dim), np.int(a_dis_dim), np.int(v_dis_dim), \
               np.int(l_dis_dim) + np.int(a_dis_dim) + np.int(v_dis_dim)

    def get_combined_dim_and_type_with_pre_proj(self):
        """
        Check and get the combination dimensionality of inputs modalities; indicate the combination type
        :return:
        """
        if self.partial_mode == 3:
            combined_dim = self.d_l + self.d_a + self.d_v
            combined_type = 'lav'
        elif self.partial_mode == 2 and not self.lonly:
            combined_dim = self.d_v + self.d_a
            combined_type = 'av'
        elif self.partial_mode == 2 and not self.aonly:
            combined_dim = self.d_l + self.d_v
            combined_type = 'lv'
        elif self.partial_mode == 2 and not self.vonly:
            combined_dim = self.d_l + self.d_a
            combined_type = 'la'
        elif self.partial_mode == 1 and self.lonly:
            combined_dim = self.d_l
            combined_type = 'l'
        elif self.partial_mode == 1 and self.aonly:
            combined_dim = self.d_a
            combined_type = 'a'
        elif self.partial_mode == 1 and self.vonly:
            combined_dim = self.d_v
            combined_type = 'v'
        else:
            raise ValueError("unknown partial mode type")
        # return combined_dim, combined_type
        return combined_dim, combined_type

    def get_fusion_network(self, self_type='cross_fusion', layers=-1):
        combined_dim, combined_type = self.get_combined_dim_and_type_with_pre_proj()
        if self_type in ['cross_fusion']:
            embed_dim, attn_dropout = combined_dim, self.attn_dropout
            print('embed_dim:', embed_dim)
        else:
            raise ValueError("unknown network type for cross fusion")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def get_network(self, self_type='l', layers=-1):
        """
        get single modality refinement module: transformer encoder network
        :param self_type:
        :param layers:
        :return:
        """
        combined_dim, combined_type = self.get_combined_dim_and_type_with_pre_proj()
        if self_type in ['l_refine', ]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['v_refine', ]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type in ['a_refine', ]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        else:
            raise ValueError("Unknown network type")

        # parameter set 1: factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,dropout=0.0
        # parameter set 2: factor=5, d_model=512, n_heads=4, e_layers=3, d_layers=2, d_ff=512,dropout=0.5
        return DistNet(enc_in=self.dis_d_model, dec_in=embed_dim, c_out=embed_dim, seq_len=None, label_len=None,
                       out_len=embed_dim, factor=self.dis_factor, d_model=self.dis_d_model, n_heads=self.dis_n_heads,
                       e_layers=self.dis_e_layers, d_layers=2, d_ff=self.dis_d_ff,
                       dropout=self.dis_dropout, attn=self.dis_attn, embed='fixed', freq='h', activation='relu',
                       output_attention=False, distil=True)

    def temp_projection(self, x_temp, x_type='l'):
        """
        projection of tensor is along the 1st dimension using conv1d; here we project timestep dimension
        :param x_temp:
        :param x_type:
        :return:
        """

        if self.temp_proj_dim == 0:
            if type(x_temp) == tuple:
                x_temp = x_temp[0]
            x_temp = x_temp[-1]
        else:
            if x_type is 'l':
                # print('x_temp:', x_temp.size())  # 32 14 30
                x_temp = self.temp_proj_l(x_temp)
            elif x_type is 'a':
                x_temp = self.temp_proj_a(x_temp)
            elif x_type is 'v':
                x_temp = self.temp_proj_v(x_temp)
            else:
                raise ValueError("unknown temp projection type!")
        return x_temp

    def sink_horn_dist(self, h_fusion):
        p_ls = c_ls = []
        lav_splits = h_fusion
        _, bs, _ = lav_splits[0].size()
        dist = lav_splits[0]
        if len(lav_splits) == 3:
            l_split = lav_splits[0].transpose(0, 1).reshape(bs, -1).unsqueeze(-1)
            a_split = lav_splits[1].transpose(0, 1).reshape(bs, -1).unsqueeze(-1)
            v_split = lav_splits[2].transpose(0, 1).reshape(bs, -1).unsqueeze(-1)
            d_01, p_01, c_01 = self.sink_horn(l_split, a_split)
            d_12, p_12, c_12 = self.sink_horn(a_split, v_split)
            d_20, p_20, c_20 = self.sink_horn(v_split, l_split)
            dist = torch.mean(d_01).add(torch.mean(d_12))
            dist = dist.add(torch.mean(d_20))
            p_ls = [p_01, p_12, p_20]
            c_ls = [c_01, c_12, c_20]
        if len(lav_splits) == 2:
            fst_split = lav_splits[0].transpose(0, 1).reshape(bs, -1).unsqueeze(-1)
            scd_split = lav_splits[1].transpose(0, 1).reshape(bs, -1).unsqueeze(-1)

            d_01, p_01, c_01 = self.sink_horn(fst_split, scd_split)
            dist = torch.mean(d_01)
            p_ls.append(p_01)
            c_ls.append(c_01)
        if len(lav_splits) == 1:
            d_01, p_01, c_01 = 0.0, 0.0, 0.0
            dist = torch.mean(torch.tensor(d_01))
            p_ls.append(p_01)
            c_ls.append(c_01)

        return dist, p_ls, c_ls

    def bert_text_feature(self, text):
        # bert_sent_mask : consists of seq_len of 1, followed by padding of 0.
        bert_sent, bert_sent_mask, bert_sent_type = text[:, 0, :], text[:, 1, :], text[:, 2, :]

        bert_output = self.bert_model(text)  # [batch_size, seq_len, 768]

        # Use the mean value of bert of the front real sentence length as the final representation of text.???why?
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        return bert_output

    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        takes input tensor size of [batch_size, input_seq_len, channels]--enc_in
                        decoder input [batch_size, output_seq_len, channels]---dec_in # note: in sentiment analysis do not
                        need decoder
        """
        combined_dim, combined_type = self.get_combined_dim_and_type_with_pre_proj()
        if self.use_bert:
            x_l = self.bert_text_feature(x_l)

        # mosi-glove:  torch.Size([32, 50, 300]) torch.Size([32, 375, 5]) torch.Size([32, 500, 20])
        # mosi-bert:  torch.Size([32, 50, 768]) torch.Size([32, 375, 5]) torch.Size([32, 500, 20])

        if not self.use_bert:
            x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout,
                            training=self.training)  # torch.Size([24, 300, 50])
        x_a = x_a.transpose(1, 2)  # torch.Size([24, 74, 500])
        x_v = x_v.transpose(1, 2)  # torch.Size([24, 35, 500])

        if not self.use_bert:
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # [batch_size, n_features, timestep]
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        if not self.use_bert:
            proj_x_l = proj_x_l.permute(0, 2, 1)
        proj_x_a = proj_x_a.permute(0, 2, 1)  # [batch_size, timestep, n_feature]
        proj_x_v = proj_x_v.permute(0, 2, 1)

        h_list = sink_emd_ls = sink_dis_ls = []

        if 'l' in list(combined_type):
            if not self.use_bert:
                h_l_refine, att_l_ls = self.l_refine(
                    proj_x_l)  # torch.Size([24, 50, 30]), , batchsize, timestep, n_feature
                h_l_refine = self.temp_projection(h_l_refine, 'l').permute(1, 0, 2)
            else:
                h_l_refine = self.bert_feature_projection(x_l.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)
                h_l_refine = h_l_refine.expand(self.temp_proj_dim, -1, -1)

            h_list.append(h_l_refine)

        if 'a' in list(combined_type):
            h_a_refine, att_a_ls = self.a_refine(proj_x_a)  # torch.Size([500, 24, 30])
            h_a_refine = self.temp_projection(h_a_refine, 'a').permute(1, 0, 2)  # torch.Size([1, 32, 30])
            h_list.append(h_a_refine)
        if 'v' in list(combined_type):
            h_v_refine, att_v_ls = self.v_refine(proj_x_v)  # torch.Size([500, 24, 30])
            h_v_refine = self.temp_projection(h_v_refine, 'v').permute(1, 0, 2)  # torch.Size([1, 32, 30])
            h_list.append(h_v_refine)

        if self.partial_mode == 3 and combined_type is 'lav':
            h_fusion = self.cross_modal_fusion(torch.cat([h_l_refine, h_a_refine, h_v_refine], dim=2))
        elif self.partial_mode == 2:
            if combined_type is 'la':
                h_fusion = self.cross_modal_fusion(torch.cat([h_l_refine, h_a_refine], dim=2))
            if combined_type is 'lv':
                h_fusion = self.cross_modal_fusion(torch.cat([h_l_refine, h_v_refine], dim=2))
            if combined_type is 'av':
                h_fusion = self.cross_modal_fusion(torch.cat([h_a_refine, h_v_refine], dim=2))
        elif self.partial_mode == 1:
            if combined_type is 'l':
                h_fusion = self.cross_modal_fusion(h_l_refine)
            if combined_type is 'a':
                h_fusion = self.cross_modal_fusion(h_a_refine)
            if combined_type is 'v':
                h_fusion = self.cross_modal_fusion(h_v_refine)

        if type(h_fusion) == tuple:
            h_fusion = h_fusion[0]

        t, bs, emb = h_fusion.size()

        d_ls, p_ls, c_ls = self.sink_horn_dist(h_list)
        h_list.append(p_ls)
        h_list.append(c_ls)
        h_list.append(d_ls)

        last_h_fusion = h_fusion.transpose(0, 1).reshape(bs, -1)  # Take the last output for prediction

        # A residual block
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_h_fusion)), p=self.out_dropout, training=self.training))

        output = self.out_layer(last_hs_proj)

        return output, last_h_fusion, h_list
