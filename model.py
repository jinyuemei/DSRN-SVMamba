"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import init
from classification.models.vmamba import *
import torch
from mamba_ssm import Mamba
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor,VisionMamba_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM,BiGRU
from modules.prediction import Attention
from modules.vitstr import create_vitstr
from modules.Vim import VisionMamba

import torch
from torch import nn

# 通道注意力模块



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction,
                       'ViTSTR': opt.Transformer,'VimSTR':opt.Vim}
        '''
        self.mambamodel=Mamba(d_model=opt.hidden_size, # Model dimension d_model
                              d_state=16,  # SSM state expansion factor
                              d_conv=4,    # Local convolution width
                              expand=2,)
        '''
        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        if opt.Vim:
            self.vimstr= VisionMamba(
                patch_size=16,
                embed_dim=192,
                depth=24,
                num_classes=opt.num_class,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                final_pool_type="mean",
                if_abs_pos_embed=True,
                if_rope=True,
                if_rope_residual=False,
                #bimamba_type="V2",
                if_cls_token=True,
                #if_devide_out=True,
                use_middle_cls_token=True)
            return

        #self.initfeatureextraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'VisionMamba':
            self.FeatureExtraction = VisionMamba_FeatureExtractor(patch_size=16,
                                                                   embed_dim=512,
                                                                   depth=24,
                                                                   rms_norm=True,
                                                                   residual_in_fp32=True,
                                                                   fused_add_norm=True,
                                                                   final_pool_type="mean",
                                                                   if_abs_pos_embed=True,
                                                                   if_rope=True,
                                                                   if_rope_residual=False,
                                                                   if_cls_token=True,
                                                                   use_middle_cls_token=True)
        elif opt.FeatureExtraction == 'VMamba':
            self.FeatureExtraction = Backbone_VSSM(out_indices=[1])
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1


        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
                BidirectionalLSTM(256, 256, 256))
            self.SequenceModeling_output = opt.hidden_size
        elif opt.SequenceModeling == 'BiGRU':
            self.SequenceModeling = BiGRU(input_size=self.FeatureExtraction_output,hidden_size=opt.hidden_size,num_layers=2,output_size=opt.hidden_size)
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True, seqlen=25):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        if self.stages['VimSTR']:
            prediction = self.vimstr(input, seqlen=seqlen)
            return prediction

        """ Feature extraction stage """
        #visual_feature = self.initfeatureextraction(input)
        visual_feature = self.FeatureExtraction(input)[0]
        visual_feature = visual_feature.permute(0,2,3,1)
        visual_feature = visual_feature.flatten(start_dim=1,end_dim=2)
        #visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        #visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
            #contextual_feature = self.mambamodel(visual_feature)
        elif self.stages['Seq'] == 'BiGRU':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        #contextual_feature = torch.cat([visual_feature,contextual_feature],2)


        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction

class JitModel(Model):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.vitstr= create_vitstr(num_tokens=opt.num_class, model=opt.TransformerModel)

    def forward(self, input, seqlen:int = 25):
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction

