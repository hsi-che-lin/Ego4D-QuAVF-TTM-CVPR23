import torch
import torch.nn as nn
import timm
from ..common.utils import countParam, freezeWeights


class TemporalModule(nn.Module):
    def __init__(self, dim, numHeads, maxLength, cls = True):
        super(TemporalModule, self).__init__()
        if (cls): maxLength += 1

        self.attn = nn.MultiheadAttention(dim, numHeads)
        self.embed = nn.Embedding(maxLength, dim)
        self.dim = dim
        self.cls = cls


    def forward(self, x):
        assert (x.dim() == 3),                                                  \
            "expect input to be a tensor of shape (T, B, D)"
        assert (x.shape[2] == self.dim),                                        \
            f"expect feature dimension to be {self.dim} (got {x.shape[2]})"

        T, B, D = x.shape
        
        if (self.cls):
            cls = torch.zeros((1, B), dtype = torch.int64, device = x.device)
            cls = self.embed(cls)
            idx = torch.arange(1, T + 1, device = x.device)
            pos = self.embed(idx).unsqueeze(1).repeat(1, B, 1)
            x = torch.cat([cls, x + pos], dim = 0)
        else:
            idx = torch.arange(0, T, device = x.device)
            pos = self.embed(idx).unsqueeze(1).repeat(1, B, 1)
            x = x + pos
        
        out, _ = self.attn(x, x, x)
        
        return out


class VisionBaseline(nn.Module):
    def __init__(self, modelType, dim, numHeads, numLayers, featLength,
                 useScore, scoreBin, freezeBackbone, dropout = 0):
        super(VisionBaseline, self).__init__()
        self.backbone = timm.create_model(
            model_name = modelType,
            pretrained = True,
            num_classes = dim
        )
        self.inProj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.temporalModule = nn.ModuleList([
            TemporalModule(dim, numHeads, featLength) for _ in range(numLayers)
        ])
        self.predHead = nn.Sequential(
            nn.Dropout1d(dropout),
            nn.Linear(2 * dim, 2) if (useScore) else nn.Linear(dim, 2)
        )
        self.dim = dim
        self.useScore = useScore
        self.scoreBin = scoreBin
        
        if (useScore):
            self.scoreEmbed = nn.Embedding(scoreBin + 1, dim)
        
        if (freezeBackbone):
            freezeWeights(self.backbone)
            
            for p in self.backbone.get_classifier().parameters():
                p.requires_grad = True


    def forward(self, feat, device):
        assert (feat["faceCrops"].dim() == 5),                                  \
            "expect video to be a tensor of shape (B, T, C, H, W) "             \
            f"but got {feat['faceCrops'].shape}"
        
        video = feat["faceCrops"].to(device)
        B, T, C, H, W = video.shape

        video = video.reshape((B * T, C, H, W))
        vFeat = self.backbone(video)
        vFeat = vFeat.reshape((B, T, self.dim)).permute(1, 0, 2)
        
        for module in self.temporalModule:
            out = module(vFeat)
            vFeat = out[1:]
            clsToken = out[0]

        if (self.useScore):
            score = (feat["score"] * self.scoreBin).to(torch.int64).to(device)
            scoreFeat = self.scoreEmbed(score)
            feat = torch.cat([clsToken, scoreFeat], dim = 1)
        else:
            feat = clsToken

        out = self.predHead(feat)

        return out


def getModel(cfg, device, verbose = True):
    model = VisionBaseline(
        modelType      = cfg.modelType,
        dim            = cfg.dim,
        numHeads       = cfg.numHeads,
        numLayers      = cfg.numLayers,
        featLength     = cfg.featLength,
        useScore       = cfg.useScore,
        scoreBin       = cfg.scoreBin,
        freezeBackbone = cfg.freezeBackbone,
        dropout        = cfg.dropout
    ).to(device)

    if (cfg.startFromCkpt):
        model.load_state_dict(torch.load(cfg.ckptLoadPath)["model"])
        print("[Model] Loading model from checkpoint")
    
    if (verbose):
        total, trainable = countParam(model)
        print(f"[Model] model parameters: {trainable}/{total}")
    
    return model


if (__name__ == "__main__"):
    class config:
        def __init__(self):
            self.modelType = "resnet50"
            self.dim = 128
            self.numHeads = 4
            self.numLayers = 1
            self.featLength = 11
            self.useScore = True
            self.scoreBin = 10
            self.freezeBackbone = True
            self.dropout = 0.2
            self.startFromCkpt = False

    cfg = config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, W = 32, 224
    score = torch.randint(0, cfg.scoreBin + 1, (B, ), device = device)
    feat = {
        "faceCrops": torch.randn((B, cfg.featLength, 3, W, W), device = device),
        "score": score / cfg.scoreBin
    }
    model = getModel(cfg, device)
    print(feat["faceCrops"].shape)
    print(feat["score"].shape)
    out = model(feat, device)
    print(out.shape)

    assert (out.shape == (B, 2))