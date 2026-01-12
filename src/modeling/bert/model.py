# """
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import code
import torch
from torch import nn
import numpy as np
from .bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from .bert import BertLayerNorm as LayerNormClass
import src.modeling.data.config as cfg
from .param_regressor import Hand_Parameter_Regressor, Face_Parameter_Regressor

def contains_nan(tensor):
    return torch.isnan(tensor).any().item()

class DICE_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(DICE_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # print(config.max_position_embeddings, "config.max_position_embeddings")
        # input()
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.position_embeddings = nn.Embedding(1024, config.hidden_size)
        self.img_dim = config.img_feature_dim 

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        # print(batch_size)
        # print(seq_length)
        # input()
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(embeddings,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs

class DICE_Module(BertPreTrainedModel):
    def __init__(self, config):
        super(DICE_Module, self).__init__(config)
        self.config = config
        self.bert = DICE_Encoder(config)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        self.apply(self.init_weights)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None):
        '''
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        '''
        predictions = self.bert(img_feats=img_feats, input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        if self.config.output_attentions and self.config.output_hidden_states:
            return pred_score, predictions[1], predictions[-1]
        else:
            return pred_score


class DICE_Network(torch.nn.Module):
    def __init__(self, args, config, backbone, trans_encoder, deform_encoder, n_hand_joints=21, n_head_joints=68, n_hand_verts=195, n_head_verts=559):
        super(DICE_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.deform_encoder = deform_encoder
        self.n_hand_joints = n_hand_joints
        self.n_head_joints = n_head_joints
        self.n_hand_verts = n_hand_verts
        self.n_head_verts = n_head_verts
        self.hand_upsampling = torch.nn.Linear(195, 778)
        self.head_upsampling1 = torch.nn.Linear(559, 1675)
        self.head_upsampling2 = torch.nn.Linear(1675, 5023)

        self.head_deform_upsampling1 = torch.nn.Linear(559, 1675)
        self.head_deform_upsampling2 = torch.nn.Linear(1675, 5023)

        self.hand_contact_upsampling = torch.nn.Linear(195, 778)
        self.head_contact_upsampling1 = torch.nn.Linear(559, 1675)
        self.head_contact_upsampling2 = torch.nn.Linear(1675, 5023)

        self.hand_param_regressor = Hand_Parameter_Regressor()
        self.face_param_regressor = Face_Parameter_Regressor()

        # self.hand_conv_learn_tokens = torch.nn.Conv1d(49,self.n_hand_joints + self.n_hand_verts,1)
        self.conv_learn_tokens = torch.nn.Conv1d(49,self.n_hand_joints + self.n_hand_verts + self.n_head_joints + self.n_head_verts,1)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(self.n_hand_joints + self.n_hand_verts + self.n_head_joints + self.n_head_verts, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)

        self.feat_downsampler = torch.nn.Linear(2051, 512)
        self.contact_feat_extractor = torch.nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.contact_classifier = torch.nn.Sequential(
            nn.Linear(49, 1),
            nn.Sigmoid()
        )

    def forward(self, images, rh_ref_kps, head_ref_kps, rh_ref_vs_sub, head_ref_vs_sub2, mvm_mask=None, is_train=False):
        batch_size = images.size(0)
        head_center = head_ref_vs_sub2.mean(dim=1, keepdim=True)
        rh_center = rh_ref_vs_sub.mean(dim=1, keepdim=True)
        head_ref_vs_sub2 = head_ref_vs_sub2 - head_center
        rh_ref_vs_sub = rh_ref_vs_sub - rh_center

        head_ref_kps = head_ref_kps - head_center
        rh_ref_kps = rh_ref_kps - rh_center

        ref_vertices = torch.cat([rh_ref_kps, head_ref_kps, rh_ref_vs_sub, head_ref_vs_sub2],dim=1)

        # extract image feature maps using a CNN backbone
        image_feat = self.backbone(images)
        image_feat_newview = image_feat.view(batch_size,2048,-1)
        image_feat_newview = image_feat_newview.transpose(1,2)
        img_tokens = self.conv_learn_tokens(image_feat_newview)

        # concatinate image feat and template mesh
        img_features = torch.cat([ref_vertices, img_tokens], dim=2)

        if is_train==True: 
            mvm_mask = mvm_mask.to(self.config.device)
            constant_tensor = torch.ones_like(img_features).cuda(self.config.device)*0.01
            img_features = img_features*mvm_mask + constant_tensor*(1-mvm_mask)   

        pred_3d_coordinates = self.trans_encoder(img_features)
        
        img_features_downsampled = self.feat_downsampler(img_features)
        deform_features = self.deform_encoder(img_features_downsampled)

        pred_3d_deforms, pred_3d_contact = deform_features[:,:,:3], deform_features[:,:,3:]

        if contains_nan(pred_3d_contact):
            print("nan in pred_contact")
        if contains_nan(pred_3d_deforms):
            print("nan in pred_deforms")
        if contains_nan(pred_3d_coordinates):
            print("nan in pred_coords")
        if contains_nan(img_features):
            print("nan in img_features")
        if contains_nan(img_features_downsampled):
            print("nan in img_features_downsampled")
        if contains_nan(images):
            print("nan in images")
        if contains_nan(ref_vertices):
            print("nan in ref_vertices")

        pred_3d_hand_joints = pred_3d_coordinates[:, :self.n_hand_joints, :]
        pred_3d_head_joints = pred_3d_coordinates[:, self.n_hand_joints:self.n_hand_joints+self.n_head_joints, :]
        pred_3d_hand_vs_sub = pred_3d_coordinates[:, self.n_hand_joints+self.n_head_joints:self.n_hand_joints+self.n_head_joints+self.n_hand_verts, :]
        pred_3d_head_vs_sub2 = pred_3d_coordinates[:, self.n_hand_joints+self.n_head_joints+self.n_hand_verts:, :]

        pred_deformations_sub2 = pred_3d_deforms[:, self.n_hand_joints+self.n_head_joints+self.n_hand_verts:, :]

        pred_3d_hand_contact_sub = pred_3d_contact[:, self.n_hand_joints+self.n_head_joints:self.n_hand_joints+self.n_head_joints+self.n_hand_verts, :]
        pred_3d_head_contact_sub2 = pred_3d_contact[:, self.n_hand_joints+self.n_head_joints+self.n_hand_verts:, :]

        # learn camera parameters
        x = self.cam_param_fc(torch.cat([pred_3d_hand_joints, pred_3d_head_joints, pred_3d_hand_vs_sub, pred_3d_head_vs_sub2], dim=1))
        x = x.transpose(1,2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1,2)
        cam_param = cam_param.squeeze(-1)

        hand_transpose = pred_3d_hand_vs_sub.transpose(1,2)
        pred_3d_hand_vs_full = self.hand_upsampling(hand_transpose)

        head_transpose = pred_3d_head_vs_sub2.transpose(1,2)
        pred_3d_head_vs_sub = self.head_upsampling1(head_transpose)
        pred_3d_head_vs_full = self.head_upsampling2(pred_3d_head_vs_sub)

        pred_3d_hand_vs_full = pred_3d_hand_vs_full.transpose(1,2)
        pred_3d_head_vs_sub = pred_3d_head_vs_sub.transpose(1,2)
        pred_3d_head_vs_full = pred_3d_head_vs_full.transpose(1,2)

        hand_contact_transpose = pred_3d_hand_contact_sub.transpose(1,2)
        pred_3d_hand_contact_full = self.hand_contact_upsampling(hand_contact_transpose)

        head_contact_transpose = pred_3d_head_contact_sub2.transpose(1,2)
        pred_3d_head_contact_sub = self.head_contact_upsampling1(head_contact_transpose)
        pred_3d_head_contact_full = self.head_contact_upsampling2(pred_3d_head_contact_sub)

        pred_3d_hand_contact_full = pred_3d_hand_contact_full.transpose(1,2)
        pred_3d_head_contact_sub = pred_3d_head_contact_sub.transpose(1,2)
        pred_3d_head_contact_full = pred_3d_head_contact_full.transpose(1,2)

        head_deform_transpose = pred_deformations_sub2.transpose(1,2)
        pred_deformations_sub1 = self.head_deform_upsampling1(head_deform_transpose)
        pred_deformations = self.head_deform_upsampling2(pred_deformations_sub1)

        pred_deformations_sub1 = pred_deformations_sub1.transpose(1,2)
        pred_deformations = pred_deformations.transpose(1,2)

        rh_betas, rh_transl, rh_rot, rh_pose = self.hand_param_regressor(pred_3d_hand_vs_full)
        face_shape, face_exp, face_pose, face_rot, face_transl = self.face_param_regressor(pred_3d_head_vs_sub)

        pred_3d_coordinates_flatten = pred_3d_coordinates.view(batch_size, -1)
        presence_feature = self.contact_feat_extractor(image_feat_newview)
        pred_contact_presence = self.contact_classifier(presence_feature.squeeze(-1))


        if is_train == False:
            print("test time, presence")
            # no deform if no contact (only in test time)
            pred_deformations = (pred_contact_presence > 0.5).float().unsqueeze(-1) * pred_deformations
        # print("no presence used")

        return cam_param, pred_3d_hand_joints, pred_3d_head_joints, pred_3d_hand_vs_sub, pred_3d_hand_vs_full, pred_3d_head_vs_sub2, pred_3d_head_vs_sub, pred_3d_head_vs_full, torch.sigmoid(pred_3d_hand_contact_sub), torch.sigmoid(pred_3d_hand_contact_full), torch.sigmoid(pred_3d_head_contact_sub2), torch.sigmoid(pred_3d_head_contact_sub), torch.sigmoid(pred_3d_head_contact_full), rh_betas, rh_transl, rh_rot, rh_pose, face_shape, face_exp, face_pose, face_rot, face_transl, pred_deformations_sub2, pred_deformations_sub1, pred_deformations, pred_contact_presence
