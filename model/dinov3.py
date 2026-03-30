import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def get_dinov3_model(name):
    REPO_DIR = '/perception/users/qiuzhongyuan/modelzoo/liufen/dinov3'
    if name == 'dinov3_vit7b16':
        weights = '/perception/users/qiuzhongyuan/modelzoo/liufen/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
        model = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=weights)
    elif name == 'dinov3_vith16plus':
        weights = '/perception/users/qiuzhongyuan/modelzoo/liufen/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
        model = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=weights)
    elif name == 'dinov3_vitl16':
        weights = '/perception/users/qiuzhongyuan/modelzoo/liufen/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights)
    elif name == 'dinov3_convnext_large':
        weights = '/perception/users/qiuzhongyuan/modelzoo/liufen/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'
        model = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=weights)
    elif name == 'dinov3_convnext_base':
        weights = '/perception/users/qiuzhongyuan/modelzoo/liufen/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth'
        model = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=weights)
    else:
        raise ValueError(f"Invalid model name: {name}")
    return model

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
def conv3x3(in_channels, out_channels, stride=1, norm_layer=partial(LayerNorm, eps=1e-6, data_format='channels_first'), activation=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
        activation() if activation is not None else nn.Identity(),
    )

def conv1x1(in_channels, out_channels, stride=1, norm_layer=partial(LayerNorm, eps=1e-6, data_format='channels_first'), activation=nn.GELU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
        activation() if activation is not None else nn.Identity(),
    )

def convtranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, norm_layer=partial(LayerNorm, eps=1e-6, data_format='channels_first'), activation=nn.GELU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
        norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
        activation() if activation is not None else nn.Identity(),
    )
class FusionBlock(nn.Module):
    def __init__(self, vit_in_channels, convnext_in_channels, upsample_in_channels, out_channels, inner_channels=512):
        super().__init__()
        self.conv_1x1 = conv1x1(vit_in_channels + convnext_in_channels + upsample_in_channels, inner_channels, norm_layer=None)
        self.conv_residual = nn.Sequential(
            conv1x1(inner_channels, inner_channels//4),
            conv3x3(inner_channels//4, inner_channels//4),
            conv1x1(inner_channels//4, inner_channels),
        )
        self.conv_1x1_2 = conv1x1(inner_channels, out_channels, norm_layer=None)
    def forward(self, vit_features, convnext_features, upsample_features):
        x = torch.cat([vit_features, convnext_features, upsample_features], dim=1)
        x = self.conv_1x1(x)
        x_residual = self.conv_residual(x)
        x = self.conv_1x1_2(x + x_residual)
        return x

class DINOv3Backbone(nn.Module):
    def __init__(self, in_channels=3, 
                 vit_model='dinov3_vith16plus', 
                 convnext_model='dinov3_convnext_large', 
                 return_class_token=True,
                 proj_channels=[256, 256, 256, 256]):
        super().__init__()
        self.vit_model = get_dinov3_model(vit_model)
        self.convnext_model = get_dinov3_model(convnext_model)
        self.vit_model.eval()
        self.convnext_model.eval()
        self.vit_model.to('cuda', dtype=torch.bfloat16)
        self.convnext_model.to('cuda', dtype=torch.bfloat16)

        vit_model_num_blocks = {
            'dinov3_vit7b16': 40,
            'dinov3_vith16plus': 32,
            'dinov3_vitl16': 24,
        }
        vit_intermediate_layers = {
            'dinov3_vit7b16': [19, 29, 39],
            'dinov3_vith16plus': [15, 23, 31],
            'dinov3_vitl16': [11, 17, 23],
        }

        intermediate_channels = {
            'dinov3_vit7b16': 4096,
            'dinov3_vith16plus': 1280,
            'dinov3_vitl16': 1024,
            'dinov3_convnext_large': [192, 384, 768, 1536], # 1/4, 1/8, 1/16, 1/32
            'dinov3_convnext_base': [128, 256, 512, 1024], # 1/4, 1/8, 1/16, 1/32
        }
        self.return_class_token = return_class_token
        self.vit_channels = intermediate_channels[vit_model]
        self.convnext_channels = intermediate_channels[convnext_model]
        self.vit_intermediate_layers_ids = vit_intermediate_layers[vit_model]

        self.vit_project_layers = nn.ModuleList([
            convtranspose2d(self.vit_channels * 2, proj_channels[0], kernel_size=6, stride=4, padding=1,norm_layer=None, activation=None),
            convtranspose2d(self.vit_channels * 2, proj_channels[1], kernel_size=4, stride=2, padding=1,norm_layer=None, activation=None),
            conv1x1(self.vit_channels * 2, proj_channels[2], norm_layer=None, activation=None),
        ])
        self.convnext_project_layers = nn.ModuleList([
            conv3x3(self.convnext_channels[i], proj_channels[i], norm_layer=None, activation=None) for i in range(4)
        ])

        self.upsample = nn.ModuleList([convtranspose2d(proj_channels[i], proj_channels[i-1], norm_layer=None) for i in range(3, 0, -1)])

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(proj_channels[i], proj_channels[i], proj_channels[i], proj_channels[i]) for i in range(3, 0, -1)
        ])

        self.channels = proj_channels[0]

    @torch.no_grad()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def get_base_model_features(self, x):
        B, C, H, W = x.shape
        vit_features = self.vit_model.get_intermediate_layers(x, n=self.vit_intermediate_layers_ids, return_class_token=self.return_class_token)
        convnext_features = self.convnext_model.get_intermediate_layers(x, n=4, return_class_token=False)
        fusion_vit_features = []
        for idx, (patch_token, class_token) in enumerate(vit_features):
            class_token = class_token.unsqueeze(1).repeat(1, patch_token.shape[1], 1)
            fusion_token = torch.cat([patch_token, class_token], dim=2) # [B, N, C*2]
            fusion_token = fusion_token.permute(0, 2, 1).contiguous().view(B, -1, H//16, W//16)
            fusion_vit_features.append(fusion_token)
        
        reshape_convnext_features = []
        for idx, convnext_feature in enumerate(convnext_features):
            scale = 2**(idx+2)
            convnext_feature = convnext_feature.permute(0, 2, 1).contiguous().view(B, -1, H//scale, W//scale)
            reshape_convnext_features.append(convnext_feature)
        return fusion_vit_features, reshape_convnext_features
    
    def forward(self, x):
        vit_features, convnext_features = self.get_base_model_features(x)
        vit_features = [self.vit_project_layers[i](vit_features[i].float()) for i in range(3)]
        convnext_features = [self.convnext_project_layers[i](convnext_features[i].float()) for i in range(4)]

        convnext_features_32x_16x = self.upsample[0](convnext_features[3])
        fusion_16x = self.fusion_blocks[0](vit_features[2], convnext_features[2], convnext_features_32x_16x)
        fusion_features_16x_8x = self.upsample[1](fusion_16x)
        fusion_8x = self.fusion_blocks[1](vit_features[1], convnext_features[1], fusion_features_16x_8x)
        fusion_features_8x_4x = self.upsample[2](fusion_8x)
        fusion_4x = self.fusion_blocks[2](vit_features[0], convnext_features[0], fusion_features_8x_4x)

        return fusion_4x
    
class Dinov3BEVNeck(nn.Module):
    def __init__(self, in_channels, out_channels, model_name='dinov3_convnext_base', **kwargs):
        super().__init__()
        self.model_name = model_name
        intermediate_channels = {
            'dinov3_convnext_large': [192, 384, 768, 1536], # 1/4, 1/8, 1/16, 1/32
            'dinov3_convnext_base': [128, 256, 512, 1024], # 1/4, 1/8, 1/16, 1/32
        }
        self.channels = intermediate_channels[model_name]
        convnext_model = get_dinov3_model(model_name)
        
        self.proj = conv1x1(in_channels, self.channels[0], activation=None)
        self.downsample_layers = convnext_model.downsample_layers[1:4]
        self.stages = convnext_model.stages

        self.upsample = nn.ModuleList([conv1x1(self.channels[0], out_channels, norm_layer=None),
                                       convtranspose2d(self.channels[1], out_channels, kernel_size=2, stride=2, padding=0, norm_layer=None),
                                       convtranspose2d(self.channels[2], out_channels, kernel_size=4, stride=4, padding=0, norm_layer=None),
                                       convtranspose2d(self.channels[3], out_channels, kernel_size=8, stride=8, padding=0, norm_layer=None),
                                       ])
        self.fusion_blocks = conv3x3(out_channels * 4, out_channels, norm_layer=None)

    def forward(self, x):
        x = self.proj(x)
        x_1x = self.stages[0](x)
        x1 = self.upsample[0](x_1x)

        x_2x = self.downsample_layers[0](x_1x)
        x_2x = self.stages[1](x_2x)
        x2 = self.upsample[1](x_2x)

        x_3x = self.downsample_layers[1](x_2x)
        x_3x = self.stages[2](x_3x)
        x3 = self.upsample[2](x_3x)

        x_4x = self.downsample_layers[2](x_3x)
        x_4x = self.stages[3](x_4x)
        x4 = self.upsample[3](x_4x)

        x = self.fusion_blocks(torch.cat([x1, x2, x3, x4], dim=1))

        return x
        