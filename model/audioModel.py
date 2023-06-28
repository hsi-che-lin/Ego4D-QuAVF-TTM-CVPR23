import torch
import torch.nn as nn
import whisper
from ..common.utils import countParam, freezeWeights


WHISPER_DIM = {
    "tiny": 384,
    "base": 512,
    "small": 768,
    "medium": 1024,
    "large": 1280
}


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


class AudioBaseline(nn.Module):
    def __init__(self, backboneType, dim, numHeads, numLayers, freezeBackbone,
                 dropout = 0):
        super(AudioBaseline, self).__init__()
        self.backbone = whisper.load_model(backboneType).encoder
        self.inProj = nn.Sequential(
            nn.Linear(WHISPER_DIM[backboneType], dim),
            nn.ReLU()
        )
        self.temporalModule = nn.ModuleList([
            TemporalModule(dim, numHeads, 1500) for _ in range(numLayers)
        ])
        self.avgPool = nn.AdaptiveAvgPool1d(900)
        self.predHead = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 2)
        )
        self.dim = dim

        if (freezeBackbone):
            freezeWeights(self.backbone)


    def forward(self, feat, device):
        assert ((feat.dim() == 3) and (feat.shape[1:] == (80, 3000))),          \
            "expect input to be a tensor of shape (B, D, T) = (B, 80, 3000)"
        
        feat = feat.to(device)
        feat = self.backbone(feat)
        feat = self.inProj(feat).permute(1, 0, 2)
        
        for module in self.temporalModule:
            feat = module(feat)[1:]
        
        feat = self.avgPool(feat.permute(1, 2, 0)).permute(0, 2, 1)
        out = self.predHead(feat).permute(0, 2, 1)
        
        return out


def getModel(cfg, device, verbose = True):
    model = AudioBaseline(
        backboneType   = cfg.backboneType,
        dim            = cfg.dim,
        numHeads       = cfg.numHeads,
        numLayers      = cfg.numLayers,
        freezeBackbone = cfg.freezeBackbone,
        dropout        = cfg.dropout
    ).to(device)

    if (cfg.startFromCkpt):
        print("[Model] loading model check points")
        model.load_state_dict(torch.load(cfg.ckptLoadPath)["model"])
    
    if (verbose):
        total, trainable = countParam(model)
        print(f"[Model] model parameters: {trainable}/{total}")
    
    return model


if (__name__ == "__main__"):
    class config:
        def __init__(self):
            self.backboneType = "tiny"
            self.dim = 128
            self.numHeads = 4
            self.numLayers = 2
            self.freezeBackbone = True
            self.startFromCkpt = False
            self.dropout = 0

    cfg = config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 32
    aud = torch.randn((B, 480000))
    mel = whisper.log_mel_spectrogram(aud)

    model = getModel(cfg, device)
    print(mel.shape)
    out = model(mel, device)
    print(out.shape)

    assert (out.shape == (B, 2, 900))