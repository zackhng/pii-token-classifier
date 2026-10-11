"""Deeper token-classification head (v5 grid, user request 2026-10-09): N extra transformer layers
between the encoder output and the classifier.

    encoder (XLM-R) -> [N x TransformerEncoderLayer(768, 12 heads, 3072 FFN, GELU, post-LN)] -> dropout -> classifier

The extra layers are randomly initialised; the classifier is the base model's own (random for a
fresh run). freeze_encoder=True trains only the extra layers + classifier (grid row H-L*), False
fine-tunes everything (row A-L*). The new layers get their own learning rate (param_groups).

Saved as: the base model (save_pretrained, so config / tokenizer load as usual) + deep_head.pt (extra
layers) + deep_head.json ({"n_layers", "freeze_encoder"}). predict.PIIPredictor loads it automatically.
"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

CONFIG_NAME, WEIGHTS_NAME = "deep_head.json", "deep_head.pt"


class DeepHeadTokenClassifier(nn.Module):
    def __init__(self, base: nn.Module, n_layers: int, freeze_encoder: bool = False):
        super().__init__()
        self.base = base
        self.config = base.config
        self.n_layers = n_layers
        self.freeze_encoder = freeze_encoder
        c = base.config
        self.extra = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=c.hidden_size, nhead=c.num_attention_heads, dim_feedforward=c.intermediate_size,
                                       dropout=c.hidden_dropout_prob, activation="gelu", batch_first=True,
                                       layer_norm_eps=c.layer_norm_eps)
            for _ in range(n_layers)])
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @property
    def encoder(self):
        return getattr(self.base, self.base.base_model_prefix)

    def new_parameters(self):
        """The randomly initialised part: extra layers + classifier (their own learning rate)."""
        return list(self.extra.parameters()) + list(self.base.classifier.parameters())

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pad = None if attention_mask is None else attention_mask == 0
        for layer in self.extra:
            h = layer(h, src_key_padding_mask=pad)
        logits = self.base.classifier(self.base.dropout(h))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)).float(), labels.view(-1), ignore_index=-100)
        return TokenClassifierOutput(loss=loss, logits=logits)

    def save_pretrained(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(out)
        torch.save(self.extra.state_dict(), out / WEIGHTS_NAME)
        (out / CONFIG_NAME).write_text(json.dumps({"n_layers": self.n_layers, "freeze_encoder": self.freeze_encoder}),
                                       encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir, dtype=torch.float32):
        d = Path(model_dir)
        cfg = json.loads((d / CONFIG_NAME).read_text(encoding="utf-8"))
        base = AutoModelForTokenClassification.from_pretrained(str(d), dtype=dtype)
        model = cls(base, cfg["n_layers"], cfg.get("freeze_encoder", False))
        model.extra.load_state_dict(torch.load(d / WEIGHTS_NAME, map_location="cpu"))
        return model


def is_deep_head_model(model_dir) -> bool:
    return (Path(model_dir) / CONFIG_NAME).exists()
