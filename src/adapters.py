"""Language adapters for Model B: one shared (frozen) encoder, a small bottleneck adapter per
language in every layer, one shared PII head.

Placement (Houlsby / Pfeiffer "output adapter"): inside each transformer layer, after the
feed-forward output projection and before the residual LayerNorm:

    h = dropout(dense(ffn))              # XLMRobertaOutput as usual
    h = h + up(gelu(down(LN(h))))         # language adapter (b = bottleneck size)
    out = LayerNorm(h + attention_out)

`up` starts at zero, so a fresh adapter is the identity and an adapter model reproduces its base
model exactly before training. Every example in a mixed-language batch goes through its own
language's adapter (rows grouped per language). zh-Hans and zh-Hant share the "zh" adapter.

Written here rather than with AdapterHub's `adapters` package, which pins older transformers.
"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForTokenClassification

# language code -> adapter name (zh-Hans / zh-Hant share one, as in the project plan)
LANG2ADAPTER = {"en": "en", "zh-Hans": "zh", "zh-Hant": "zh", "ja": "ja", "ko": "ko", "hi": "hi", "ar": "ar",
                "th": "th", "vi": "vi", "ms": "ms", "id": "id", "tl": "tl"}
ADAPTERS = sorted(set(LANG2ADAPTER.values()))
LANG_IDS = {lang: ADAPTERS.index(a) for lang, a in LANG2ADAPTER.items()}   # lang code -> adapter index
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapters.pt"


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.down = nn.Linear(hidden, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class _Route:
    """Per-forward routing state shared by all layers: adapter index per example."""
    lang_ids: torch.Tensor | None = None


class AdapterOutput(nn.Module):
    """Drop-in replacement for XLMRobertaOutput (same dense / LayerNorm / dropout parameters, so
    state-dict keys of the base model are unchanged) with a bank of language adapters."""

    def __init__(self, orig, bottleneck: int, route: _Route):
        super().__init__()
        self.dense, self.LayerNorm, self.dropout = orig.dense, orig.LayerNorm, orig.dropout
        hidden = orig.dense.out_features
        self.adapters = nn.ModuleList([BottleneckAdapter(hidden, bottleneck) for _ in ADAPTERS])
        self.route = route

    def forward(self, hidden_states, input_tensor):
        h = self.dropout(self.dense(hidden_states))
        ids = self.route.lang_ids
        if ids is None:
            raise RuntimeError("adapter model called without lang_ids")
        out = torch.empty_like(h)
        for a in ids.unique().tolist():
            rows = ids == a
            out[rows] = self.adapters[a](h[rows])
        return self.LayerNorm(out + input_tensor)


class AdapterTokenClassifier(nn.Module):
    """Token classifier with per-language adapters. forward(..., lang_ids) - lang_ids is a
    LongTensor [batch] of adapter indices (LANG_IDS)."""

    def __init__(self, base: nn.Module, bottleneck: int = 64, freeze_base: bool = True):
        super().__init__()
        self.base = base
        self.config = base.config
        self.bottleneck = bottleneck
        self.route = _Route()
        encoder = getattr(base, base.base_model_prefix).encoder
        for layer in encoder.layer:
            layer.output = AdapterOutput(layer.output, bottleneck, self.route)
        if freeze_base:
            for name, p in self.named_parameters():
                p.requires_grad = ".adapters." in name or name.startswith("base.classifier.")

    def forward(self, input_ids=None, attention_mask=None, labels=None, lang_ids=None, **kw):
        self.route.lang_ids = lang_ids
        try:
            return self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kw)
        finally:
            self.route.lang_ids = None

    # -------------------------------------------------------------- parameter accounting
    def adapter_params(self) -> dict:
        per = {a: 0 for a in ADAPTERS}
        for name, p in self.named_parameters():
            if ".adapters." in name:
                per[ADAPTERS[int(name.split(".adapters.")[1].split(".")[0])]] += p.numel()
        return per

    # -------------------------------------------------------------- save / load
    def save_pretrained(self, out_dir, base_dir: str | None = None):
        """Adapters + head always; the (frozen) encoder only if `base_dir` is not given, i.e. it
        was fine-tuned. Config records which base to load."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        trainable = {k: v for k, v in self.state_dict().items()
                     if ".adapters." in k or k.startswith("base.classifier.")}
        torch.save(trainable, out / WEIGHTS_NAME)
        if base_dir is None:
            self.base.save_pretrained(out / "base")
        self.config.save_pretrained(out)
        with open(out / CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump({"bottleneck": self.bottleneck, "adapters": ADAPTERS, "lang2adapter": LANG2ADAPTER,
                       "base": base_dir or "base"}, f, indent=1)

    @classmethod
    def from_pretrained(cls, model_dir, dtype=torch.float32):
        d = Path(model_dir)
        acfg = json.loads((d / CONFIG_NAME).read_text(encoding="utf-8"))
        base_path = d / acfg["base"] if (d / acfg["base"]).exists() else acfg["base"]
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(d)
        base = AutoModelForTokenClassification.from_pretrained(str(base_path), config=config, dtype=dtype)
        model = cls(base, acfg["bottleneck"], freeze_base=True)
        missing, unexpected = model.load_state_dict(torch.load(d / WEIGHTS_NAME, map_location="cpu"), strict=False)
        assert not unexpected, unexpected
        assert not [k for k in missing if ".adapters." in k or "classifier" in k], missing
        return model


def is_adapter_model(model_dir) -> bool:
    return (Path(model_dir) / CONFIG_NAME).exists()
