import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 

class MultiConv_Transformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=20, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp,
                 dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, **kwargs):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        
        # Pass through convolutional layers
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # Patch embedding for the original input
        x_original = self.patch_embed(x)

        # Padding the conv results to match the size of original patches
        pad_size = x_original.shape[-2:]
        x1 = F.pad(x1, (0, pad_size[1] - x1.shape[-1], 0, pad_size[0] - x1.shape[-2]))
        x2 = F.pad(x2, (0, pad_size[1] - x2.shape[-1], 0, pad_size[0] - x2.shape[-2]))
        x3 = F.pad(x3, (0, pad_size[1] - x3.shape[-1], 0, pad_size[0] - x3.shape[-2]))
        
        # Patch embedding for each stage
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        x3 = self.patch_embed(x3)

        # Add class tokens and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_original = torch.cat((cls_tokens, x_original), dim=1)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x3 = torch.cat((cls_tokens, x3), dim=1)
        x_original = x_original + self.pos_embed[:, :x_original.size(1), :]
        x1 = x1 + self.pos_embed[:, :x1.size(1), :]
        x2 = x2 + self.pos_embed[:, :x2.size(1), :]
        x3 = x3 + self.pos_embed[:, :x3.size(1), :]

        # Pass through attention blocks
        for blk in self.blocks:
            x_original = blk(x_original)
            x1 = blk(x1)
            x2 = blk(x2)
            x3 = blk(x3)

        # Norm layers
        x_original = self.norm(x_original)
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        x3 = self.norm(x3)

        # Extract class token from each stage
        x_original = x_original[:, 0]
        x1 = x1[:, 0]
        x2 = x2[:, 0]
        x3 = x3[:, 0]

        # Average pooling of the class tokens from each stage
        x = (x_original + x1 + x2 + x3) / 4
        return x
        
    def forward(self, x):
        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x


# No Padding
# def forward_features(self, x):
#     B = x.shape[0]
    
#     # Pass through convolutional layers
#     x1 = self.conv1(x)
#     x2 = self.conv2(x1)
#     x3 = self.conv3(x2)
    
#     # Patch embedding for the original input
#     x_original = self.patch_embed(x)
#     x1 = self.patch_embed(x1)
#     x2 = self.patch_embed(x2)
#     x3 = self.patch_embed(x3)

#     # Add class tokens and position embeddings
#     cls_tokens = self.cls_token.expand(B, -1, -1)
#     x_original = torch.cat((cls_tokens, x_original), dim=1)
#     x1 = torch.cat((cls_tokens, x1), dim=1)
#     x2 = torch.cat((cls_tokens, x2), dim=1)
#     x3 = torch.cat((cls_tokens, x3), dim=1)
#     x_original = x_original + self.pos_embed[:, :x_original.size(1), :]
#     x1 = x1 + self.pos_embed[:, :x1.size(1), :]
#     x2 = x2 + self.pos_embed[:, :x2.size(1), :]
#     x3 = x3 + self.pos_embed[:, :x3.size(1), :]

#     # Pass through attention blocks
#     for blk in self.blocks:
#         x_original = blk(x_original)
#         x1 = blk(x1)
#         x2 = blk(x2)
#         x3 = blk(x3)

#     # Norm layers
#     x_original = self.norm(x_original)
#     x1 = self.norm(x1)
#     x2 = self.norm(x2)
#     x3 = self.norm(x3)

#     # Extract class token from each stage
#     x_original = x_original[:, 0]
#     x1 = x1[:, 0]
#     x2 = x2[:, 0]
#     x3 = x3[:, 0]

#     # Average pooling of the class tokens from each stage
#     x = (x_original + x1 + x2 + x3) / 4
#     return x
    
# def forward(self, x):
#     x = self.forward_features(x)
    
#     if self.dropout_rate:
#         x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
#     x = self.head(x)
    
#     return x
