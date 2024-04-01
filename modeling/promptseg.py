from functools import partial
from typing import Any, Mapping
from .image_encoder import ImageEncoderViT
from .image_decoder import GPT2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PromptSeg(nn.Module):
    def __init__(self, config, layernorm=torch.nn.LayerNorm, attention_mask=None):
        super().__init__()
        self.image_encoder = ImageEncoderViT(
                depth=config.MODEL.ENCODER.DEPTH,
                embed_dim=config.MODEL.ENCODER.HIDDEN_SIZE,
                img_size=config.DATA.IMG_SIZE,
                norm_layer=partial(layernorm, eps=config.MODEL.ENCODER.NORM_EPS),
                num_heads=config.MODEL.ENCODER.NUM_HEADS,
                patch_size=config.MODEL.ENCODER.PATCH_SIZE,
                use_rel_pos=config.MODEL.ENCODER.RELATIVE_POSITION_EMBEDDING,
                global_attn_indexes=config.MODEL.ENCODER.GLOBAL_ATTENTION_INDEX,
                window_size=config.MODEL.ENCODER.WINDOW_SIZE,
                out_chans=config.MODEL.ENCODER.OUT_CHANS,
                adapter_train=config.MODEL.ENCODER.ADAPTER,
                in_chans=config.DATA.IMAGE_CHANNEL,
        )
        self.image_decoder = GPT2Model(
                in_chans=config.MODEL.DECODER.INPUT_CHANS, 
                depth=config.MODEL.DECODER.DEPTH , 
                hidden_size=config.MODEL.DECODER.HIDDEN_SIZE, 
                ffn_hidden_size=config.MODEL.DECODER.FFN_HIDDEN_SIZE, 
                mlp_ratio=config.MODEL.DECODER.MLP_RATIO, 
                num_attention_heads=config.MODEL.DECODER.NUM_HEADS, 
                scale_attention_weights=config.MODEL.DECODER.SCALE_ATTENTION_WEIGHTS, 
                max_position_embeddings=config.MODEL.DECODER.MAX_POSITION_EMBEDDINGS, 
                embedding_drop_rate=config.MODEL.DECODER.EMBEDDDINGS_DROP_RATE, 
                attention_dropout_rate=config.MODEL.DECODER.ATTENTION_DROPOUT_RATE,
                hidden_dropout_rate=config.MODEL.DECODER.HIDDEN_DROPOUT_RATE, 
                norm_eps=config.MODEL.DECODER.NORM_EPS,
                attention_mask=attention_mask)
        self.neck = nn.Linear(config.MODEL.DECODER.HIDDEN_SIZE, config.MODEL.DECODER.INPUT_CHANS)
        self.head = nn.Sequential(
            nn.Linear(1, config.MODEL.HEAD.HIDDEN_SIZE),
            nn.GELU(),                  
            nn.Linear(config.MODEL.HEAD.HIDDEN_SIZE, 2),
        )
        
        self.patch_size = config.MODEL.ENCODER.PATCH_SIZE
        
        self.add_sep_token = config.MODEL.ADD_SEP_TOKEN
 

    def forward(self, images, attention_mask=None):
        '''
        input
            images: (batch_size, image_num_per_sequence, C, H, W)
            attention_mask: (batch_size, image_num_per_sequence)
        output
            image_features: (batch_size, image_num_per_sequence * patch_num_H * patch_num_W , img_out_chans)
            output_features: (batch_size, image_num_per_sequence * patch_num_H * patch_num_W , img_out_chans)
            output_pixel: (batch_size, image_num_per_sequence, C, H, W)
        '''
        batch_size = images.shape[0]
        image_num = images.shape[1]
        H = images.shape[-2]
        W = images.shape[-1]
        h,w = H//self.patch_size, W//self.patch_size
        images = rearrange(images, "b n c h w -> (b n) c h w")
        
        encoded_features = self.image_encoder(images)
        encoded_features = rearrange(encoded_features, "... c h w -> ... h w c")

        if self.add_sep_token:
            # TODO: add sep token at the end of each pair of images
            pass
        
        encoded_features = rearrange(encoded_features, "(b n) h w c -> b (n h w) c", b=batch_size, n=image_num)
        
        # attention mask
        output = self.image_decoder(inputs_embeds=encoded_features, attention_mask=attention_mask, change_dimension=True)
        output_features = self.neck(output[0])
        output_pixel = self.head(rearrange(output_features, 
                                           "b (n h w) (hp wp c) -> b n (h hp) (w wp) c",
                                           n=image_num, h=h, w=w, hp=self.patch_size, wp=self.patch_size))
        output_pixel = rearrange(output_pixel, "b n h w c -> b n c h w")
        
        return  encoded_features, output_features, output_pixel
    
        