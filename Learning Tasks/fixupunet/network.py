from __future__ import absolute_import

from torch import nn
import torch
from .modules import FixupConvModule, FixupResidualChain, FixupTuningBlock


class FixUpUnet(nn.Module):
    """
    Unet using residual blocks and residual chains without any normalization layer.
    Example of cfg to instanciate the network:

    from omegaconf import DictConfig
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 5,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": True,
        }
    )


    """

    def __init__(self, cfg):
        super(FixUpUnet, self).__init__()

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1
        norm = "none"

        layer = FixupConvModule(cfg.in_feat, cfg.feat, 3, 1, True, norm, cfg.act_fn)
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.in_conv = layer

        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            # Residual chain
            layer = FixupResidualChain(
                feat_curr,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

            # Downsampling convolution
            layer = FixupConvModule(
                feat_curr, feat_next, 4, 2, True, norm, "none", use_bias=True
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

        self.bottleneck_layers = nn.ModuleList()
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            cfg.bottleneck_layers,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers.append(layer)

        self.up_layers = nn.ModuleList()
        for i in range(cfg.down_layers, 0, -1):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i - 1) * feat, max_feat)
            # Upsample
            self.up_layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            # Merge skip and upsample
            if self.skip:
                layer = FixupConvModule(
                    feat_next + feat_curr,
                    feat_next,
                    1,
                    1,
                    False,
                    norm,
                    "none",
                    use_bias=True,
                )
                if cfg.script_submodules:
                    layer = torch.jit.script(layer)
                self.up_layers.append(layer)
            # Residual chain
            layer = FixupResidualChain(
                feat_next,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i - 1 < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.up_layers.append(layer)

        layer = FixupConvModule(
            feat, cfg.out_feat, 3, 1, True, norm, cfg.out_act_fn, use_bias=True
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.out_conv = layer

    def forward(self, x):

        skips = []
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            if i % 2 == 0:
                skips.append(x)

        for layer in self.bottleneck_layers:
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x)

            if self.skip:
                if i % 3 == 0:
                    x = torch.cat([x, skips.pop()], dim=1)

        return self.out_conv(x)

class FixUpUnetInject(nn.Module):
    """
    Unet using residual blocks and residual chains without any normalization layer.
    Example of cfg to instanciate the network:

    from omegaconf import DictConfig
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 5,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": True,
        }
    )


    """

    def __init__(self, cfg):
        super(FixUpUnetInject, self).__init__()

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1

        layer = FixupConvModule(cfg.in_feat + 1, cfg.feat, 3, 1, True, "none", cfg.act_fn)
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.in_conv = layer

        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            # Residual chain
            layer = FixupResidualChain(
                feat_curr,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

            # Downsampling convolution
            layer = FixupConvModule(
                feat_curr+1, feat_next, 4, 2, True, "none", "none", use_bias=True
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

        self.bottleneck_layers = nn.ModuleList()
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            cfg.bottleneck_layers,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers.append(layer)

        self.up_layers = nn.ModuleList()
        for i in range(cfg.down_layers, 0, -1):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i - 1) * feat, max_feat)
            # Upsample
            self.up_layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            # Merge skip and upsample
            if self.skip:
                layer = FixupConvModule(
                    feat_next + feat_curr,
                    feat_next,
                    1,
                    1,
                    False,
                    "none",
                    "none",
                    use_bias=True,
                )
                if cfg.script_submodules:
                    layer = torch.jit.script(layer)
                self.up_layers.append(layer)
            # Residual chain
            layer = FixupResidualChain(
                feat_next,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i - 1 < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.up_layers.append(layer)

        layer = FixupConvModule(
            feat, cfg.out_feat, 3, 1, True, "none", cfg.out_act_fn, use_bias=True
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.out_conv = layer

    def expandEV(self, x, ev):
        ev_map = torch.unsqueeze(ev, 1)
        ev_map = torch.unsqueeze(ev_map, 2)
        ev_map = torch.unsqueeze(ev_map, 3)
        ev_map = ev_map.expand((x.size(dim=0),1,x.size(dim=2),x.size(dim=3)))
        x = torch.cat((x, ev_map), dim=1)
        return x

    def forward(self, x, ev):

        skips = []
        x = self.expandEV(x, ev)
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            if i % 2 != 0:
                x = self.expandEV(x, ev)
                
            x = layer(x)

            if i % 2 == 0:
                skips.append(x)

        for layer in self.bottleneck_layers:
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x)

            if self.skip:
                if i % 3 == 0:
                    x = torch.cat([x, skips.pop()], dim=1)

        return self.out_conv(x)


class FixUpUnetScale(nn.Module):
    """
    Unet using residual blocks and residual chains without any normalization layer.
    Example of cfg to instanciate the network:

    from omegaconf import DictConfig
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 5,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": True,
            "input_sizex": 128,
            "input_sizey": 64,
        }
    )


    """

    def __init__(self, cfg):
        super(FixUpUnetScale, self).__init__()

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1
        norm = "none"

        layer = FixupConvModule(cfg.in_feat, cfg.feat, 3, 1, True, norm, cfg.act_fn)
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.in_conv = layer

        curr_sizex = cfg.input_sizex
        curr_sizey = cfg.input_sizey

        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            curr_sizex = curr_sizex // 2
            curr_sizey = curr_sizey // 2
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            # Residual chain
            layer = FixupResidualChain(
                feat_curr,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

            # Downsampling convolution
            layer = FixupConvModule(
                feat_curr, feat_next, 4, 2, True, norm, "none", use_bias=True
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

        self.bottleneck_layers1 = nn.ModuleList()
        bottleneck_layers1_num = cfg.bottleneck_layers//2
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            bottleneck_layers1_num,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers1.append(layer)

        #scale subnetwork
        insize = feat_curr * curr_sizex * curr_sizey
        self.fc_layers = nn.ModuleList()

        fc1 = nn.Linear(insize, 1024)
        fc2 = nn.Linear(1024, 128)
        fc3 = nn.Linear(128, 16)
        fc4 = nn.Linear(16, 1)
        self.fc_layers.append(fc1)
        self.fc_layers.append(fc2)
        self.fc_layers.append(fc3)
        self.fc_layers.append(fc4)

        self.bottleneck_layers2 = nn.ModuleList()
        bottleneck_layers2_num = cfg.bottleneck_layers - cfg.bottleneck_layers//2
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            bottleneck_layers2_num,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers2.append(layer)


        self.up_layers = nn.ModuleList()
        for i in range(cfg.down_layers, 0, -1):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i - 1) * feat, max_feat)
            # Upsample
            self.up_layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            # Merge skip and upsample
            if self.skip:
                layer = FixupConvModule(
                    feat_next + feat_curr,
                    feat_next,
                    1,
                    1,
                    False,
                    norm,
                    "none",
                    use_bias=True,
                )
                if cfg.script_submodules:
                    layer = torch.jit.script(layer)
                self.up_layers.append(layer)
            # Residual chain
            layer = FixupResidualChain(
                feat_next,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i - 1 < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.up_layers.append(layer)

        layer = FixupConvModule(
            feat, cfg.out_feat, 3, 1, True, norm, cfg.out_act_fn, use_bias=True
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.out_conv = layer

    def forward(self, x):

        skips = []
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            if i % 2 == 0:
                skips.append(x)

        for layer in self.bottleneck_layers1:
            x = layer(x)

        scale = torch.clone(x)
        scale = torch.flatten(scale,1)
        for layer in self.fc_layers:
            scale = layer(scale)
            scale = nn.functional.tanh(scale)

        for layer in self.bottleneck_layers2:
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x)

            if self.skip:
                if i % 3 == 0:
                    x = torch.cat([x, skips.pop()], dim=1)

        return self.out_conv(x), scale


class FixUpUnetChopped(nn.Module):
    """
    Unet using residual blocks and residual chains without any normalization layer.
    Example of cfg to instanciate the network:

    from omegaconf import DictConfig
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 5,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": True,
            "input_sizex": 128,
            "input_sizey": 64,
        }
    )


    """

    def __init__(self, cfg):
        super(FixUpUnetChopped, self).__init__()

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1
        norm = "none"

        layer = FixupConvModule(cfg.in_feat, cfg.feat, 3, 1, True, norm, cfg.act_fn)
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.in_conv = layer

        curr_sizex = cfg.input_sizex
        curr_sizey = cfg.input_sizey

        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            curr_sizex = curr_sizex // 2
            curr_sizey = curr_sizey // 2
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            # Residual chain
            layer = FixupResidualChain(
                feat_curr,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

            # Downsampling convolution
            layer = FixupConvModule(
                feat_curr, feat_next, 4, 2, True, norm, "none", use_bias=True
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

        self.bottleneck_layers1 = nn.ModuleList()
        bottleneck_layers1_num = cfg.bottleneck_layers//2
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            bottleneck_layers1_num,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers1.append(layer)

        #scale subnetwork
        insize = feat_curr * curr_sizex * curr_sizey
        self.fc_layers = nn.ModuleList()

        fc1 = nn.Linear(insize, 1024)
        fc2 = nn.Linear(1024, 128)
        fc3 = nn.Linear(128, 16)
        fc4 = nn.Linear(16, 1)
        self.fc_layers.append(fc1)
        self.fc_layers.append(fc2)
        self.fc_layers.append(fc3)
        self.fc_layers.append(fc4)

    def forward(self, x):

        skips = []
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            if i % 2 == 0:
                skips.append(x)

        for layer in self.bottleneck_layers1:
            x = layer(x)

        scale = torch.clone(x)
        scale = torch.flatten(scale,1)
        for layer in self.fc_layers[:-1]:
            scale = layer(scale)
            scale = nn.functional.relu(scale)
        scale = self.fc_layers[-1](scale)
        scale = nn.functional.tanh(scale)
        
        return scale


class FixUpUnetChoppedBins(nn.Module):
    """
    Unet using residual blocks and residual chains without any normalization layer.
    Example of cfg to instanciate the network:

    from omegaconf import DictConfig
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 5,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": True,
            "input_sizex": 128,
            "input_sizey": 64,
            "nbins": 100,
        }
    )


    """

    def __init__(self, cfg):
        super(FixUpUnetChoppedBins, self).__init__()

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1
        norm = "none"

        layer = FixupConvModule(cfg.in_feat, cfg.feat, 3, 1, True, norm, cfg.act_fn)
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.in_conv = layer

        curr_sizex = cfg.input_sizex
        curr_sizey = cfg.input_sizey


        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            curr_sizex = curr_sizex // 2
            curr_sizey = curr_sizey // 2
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            # Residual chain
            layer = FixupResidualChain(
                feat_curr,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

            # Downsampling convolution
            layer = FixupConvModule(
                feat_curr, feat_next, 4, 2, True, norm, "none", use_bias=True
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

        self.bottleneck_layers1 = nn.ModuleList()
        bottleneck_layers1_num = cfg.bottleneck_layers//2
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            bottleneck_layers1_num,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers1.append(layer)

        #scale subnetwork
        insize = feat_curr * curr_sizex * curr_sizey
        self.fc_layers = nn.ModuleList()

        fc1 = nn.Linear(insize, 1024)
        fc2 = nn.Linear(1024, cfg.nbins)
        self.fc_layers.append(fc1)
        self.fc_layers.append(fc2)

    def forward(self, x):

        skips = []
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            if i % 2 == 0:
                skips.append(x)

        for layer in self.bottleneck_layers1:
            x = layer(x)

        scale = torch.flatten(x,1)
            
        for layer in self.fc_layers[:-1]:
            scale = layer(scale)
            scale = nn.functional.relu(scale)
        scale = self.fc_layers[-1](scale)
        
        return scale