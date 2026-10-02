"""Domain-adaptive continued pretraining of deberta-v3-xsmall with replaced-token detection (RTD).

Follows Microsoft's DeBERTa reference (DeBERTa/apps/tasks/rtd_task.py):
- generator (released `pytorch_model.generator.bin`, 6 layers) does MLM on 15% of tokens
  (80% [MASK] / 10% random / 10% kept); its top-1 predictions replace those tokens
- discriminator (the 12-layer model we fine-tune) + released RTD head (`mask_predictions.*`)
  predicts for every token whether it was replaced
- gradient-disentangled embedding sharing (GDES): disc word embeddings = stopgrad(E_gen) + delta
- loss = MLM + rtd_lambda * RTD

Input: packed 512-token sequences from build_pretrain_corpus.py (data/pretrain/*.npy).
Output: a checkpoint every `checkpoint_every_tokens` under outputs/dapt-v3/tokens-XXXM/ with the
discriminator in HF format (loadable by AutoModelForTokenClassification) + state for --resume.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import (AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Model,
                          get_linear_schedule_with_warmup)
from transformers.activations import ACT2FN

from common import ROOT, load_yaml


# ---------------------------------------------------------------- model pieces
class RTDHead(nn.Module):
    """DeBERTa's LMMaskPredictionHead: LayerNorm([CLS] + h) -> dense -> act -> classifier."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden):
        x = self.LayerNorm(hidden[:, :1, :] + hidden)
        return self.classifier(self.act(self.dense(x))).squeeze(-1)


class SharedEmbedding(nn.Module):
    """GDES: weight = stopgrad(generator embedding) + delta (delta trained by the disc only)."""

    def __init__(self, gen_embedding: nn.Embedding, delta: torch.Tensor):
        super().__init__()
        self.gen = [gen_embedding]          # list: do not register the generator's module twice
        self.delta = nn.Parameter(delta)

    @property
    def weight(self):
        return self.gen[0].weight.detach() + self.delta

    def forward(self, ids):
        return F.embedding(ids, self.weight)


def load_generator(repo: str) -> DebertaV2ForMaskedLM:
    """Released generator; old head names lm_predictions.lm_head.* -> cls.predictions.*."""
    cfg = DebertaV2Config(**json.load(open(hf_hub_download(repo, "generator_config.json"))))
    sd = torch.load(hf_hub_download(repo, "pytorch_model.generator.bin"), map_location="cpu",
                    weights_only=True)
    ren = {"lm_predictions.lm_head.dense.": "cls.predictions.transform.dense.",
           "lm_predictions.lm_head.LayerNorm.": "cls.predictions.transform.LayerNorm.",
           "lm_predictions.lm_head.bias": "cls.predictions.bias"}
    out = {}
    for k, v in sd.items():
        for a, b in ren.items():
            if k.startswith(a):
                k = b + k[len(a):]
                break
        out[k] = v
    out["cls.predictions.decoder.weight"] = out["deberta.embeddings.word_embeddings.weight"]
    out["cls.predictions.decoder.bias"] = out["cls.predictions.bias"]
    out.pop("deberta.embeddings.position_embeddings.weight", None)   # unused (relative pos.)
    gen = DebertaV2ForMaskedLM(cfg)
    missing, unexpected = gen.load_state_dict(out, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    gen.tie_weights()
    return gen


def load_discriminator(repo: str) -> tuple[DebertaV2Model, RTDHead]:
    disc = DebertaV2Model.from_pretrained(repo, dtype=torch.float32)
    head = RTDHead(disc.config)
    sd = torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu",
                    weights_only=True)
    head.load_state_dict({k.removeprefix("mask_predictions."): v for k, v in sd.items()
                          if k.startswith("mask_predictions.")})
    return disc, head


class RTDModel(nn.Module):
    def __init__(self, gen, disc, head):
        super().__init__()
        self.gen, self.disc, self.head = gen, disc, head
        w_d = disc.embeddings.word_embeddings.weight.data.clone()
        g_emb = gen.deberta.embeddings.word_embeddings
        disc.embeddings.word_embeddings = SharedEmbedding(g_emb, w_d - g_emb.weight.data)

    def disc_logits(self, ids, attn):
        return self.head(self.disc(input_ids=ids, attention_mask=attn).last_hidden_state)

    def export_discriminator(self) -> DebertaV2Model:
        """Plain DebertaV2Model with the shared embedding materialised (for fine-tuning)."""
        d = DebertaV2Model(self.disc.config)
        sd = {k: v for k, v in self.disc.state_dict().items() if "word_embeddings" not in k}
        sd["embeddings.word_embeddings.weight"] = self.disc.embeddings.word_embeddings.weight.detach().clone()
        d.load_state_dict(sd, strict=True)
        return d


# ---------------------------------------------------------------- masking
def mask_tokens(ids, special, vocab_size, mask_id, gen: torch.Generator, p=0.15,
                p_mask=0.8, p_keep=0.1):
    """BERT-style masking. Returns (gen_input, labels) with labels = original id at the
    selected positions, -100 elsewhere. Special tokens are never selected."""
    probs = torch.full(ids.shape, p, device=ids.device)
    probs[special] = 0.0
    sel = torch.bernoulli(probs, generator=gen).bool()
    labels = torch.where(sel, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    inp = ids.clone()
    inp[sel & (r < p_mask)] = mask_id
    rnd = sel & (r >= p_mask) & (r < 1 - p_keep)
    inp[rnd] = torch.randint(vocab_size, ids.shape, device=ids.device, generator=gen)[rnd]
    return inp, labels


def rtd_step(model, ids, special_ids, mask_id, vocab_size, rng, rtd_lambda):
    """One forward pass: returns (loss, mlm_loss, rtd_loss, stats dict)."""
    attn = torch.ones_like(ids)
    special = torch.isin(ids, special_ids)
    gen_in, labels = mask_tokens(ids, special, vocab_size, mask_id, rng)
    sel = labels != -100
    # vocabulary logits only at the ~15% masked positions (same loss as the full MLM head, ~7x
    # less memory than [batch, 512, 128k] logits)
    hidden = model.gen.deberta(input_ids=gen_in, attention_mask=attn).last_hidden_state
    logits_g = model.gen.cls(hidden[sel])
    mlm_loss = F.cross_entropy(logits_g.float(), ids[sel])
    disc_in = ids.clone()
    disc_in[sel] = logits_g.detach().argmax(-1)            # reference: top-1 replacement
    rtd_labels = (sel & (disc_in != ids)).float()
    logits = model.disc_logits(disc_in, attn)
    keep = ~special
    rtd_loss = F.binary_cross_entropy_with_logits(logits[keep].float(), rtd_labels[keep])
    loss = mlm_loss + rtd_lambda * rtd_loss
    with torch.no_grad():
        p = (logits[keep] > 0).float()
        y = rtd_labels[keep]
        tp = (p * y).sum().item()
        stats = {"tp": tp, "fp": (p * (1 - y)).sum().item(), "fn": ((1 - p) * y).sum().item(),
                 "correct": (p == y).sum().item(), "n": y.numel(),
                 "replaced": y.sum().item(), "masked": sel.sum().item()}
    return loss, mlm_loss.detach(), rtd_loss.detach(), stats


def summarise(stats: dict) -> dict:
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"rtd_acc": stats["correct"] / max(1, stats["n"]),
            "rtd_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "replaced_rate": stats["replaced"] / max(1, stats["masked"])}


# ---------------------------------------------------------------- eval / probes
PROBES = ["Please enter your date of [MASK] below.",
          "Your account [MASK] is 0123-456789-0.",
          "We completed customer [MASK] diligence before onboarding the client.",
          "The client's risk [MASK] was assessed as moderate.",
          "The relationship [MASK] reviewed the portfolio with the client.",
          "The fund charges an annual management [MASK] of 1.5%."]


@torch.no_grad()
def evaluate(model, heldout, domains, names, special_ids, mask_id, vocab, device, rtd_lambda,
             batch=32, max_seqs=256):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(1234)
    res = {}
    for d, name in enumerate(names):
        idx = np.nonzero(domains == d)[0][:max_seqs]
        if not len(idx):
            continue
        agg, mlm, rtd, nb = {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "n": 0, "replaced": 0,
                             "masked": 0}, 0.0, 0.0, 0
        for i in range(0, len(idx), batch):
            ids = torch.from_numpy(heldout[idx[i:i + batch]].astype(np.int64)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, m, r, s = rtd_step(model, ids, special_ids, mask_id, vocab, rng, rtd_lambda)
            mlm, rtd, nb = mlm + m.item(), rtd + r.item(), nb + 1
            for k in agg:
                agg[k] += s[k]
        res[name] = {"mlm_loss": mlm / nb, "rtd_loss": rtd / nb, **summarise(agg)}
    model.train()
    return res


@torch.no_grad()
def probes(model, tok, device):
    model.eval()
    out = {}
    for s in PROBES:
        enc = tok(s, return_tensors="pt").to(device)
        i = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0].item()
        top = model.gen(**enc).logits[0, i].topk(5).indices.tolist()
        out[s] = [t.lstrip("▁") for t in tok.convert_ids_to_tokens(top)]
    model.train()
    return out


# ---------------------------------------------------------------- checkpointing
def save_checkpoint(path: Path, model, tok, opt, sched, state: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.export_discriminator().save_pretrained(path)
    tok.save_pretrained(path)
    torch.save({"gen": model.gen.state_dict(), "delta": model.disc.embeddings.word_embeddings.delta,
                "head": model.head.state_dict(), "disc": model.disc.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
               path / "rtd_state.pt")
    (path / "pretrain_state.json").write_text(json.dumps(state, indent=2))
    print(f"[ckpt] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", help="checkpoint dir to resume from")
    ap.add_argument("--max_tokens", type=float, help="override total tokens (e.g. 2e6 smoke test)")
    ap.add_argument("--checkpoint_every", type=float, help="override checkpoint interval (tokens)")
    ap.add_argument("--out_dir", help="override output dir")
    args = ap.parse_args()

    cfg = load_yaml("pretrain.yaml")
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / cfg["corpus"]["out_dir"]
    out_dir = ROOT / (args.out_dir or t["out_dir"])
    meta = json.loads((data_dir / "meta.json").read_text())
    seq_len = meta["seq_len"]
    train = np.load(data_dir / "train_ids.npy", mmap_mode="r")
    train_dom = np.load(data_dir / "train_domain_tokens.npy", mmap_mode="r")
    heldout = np.load(data_dir / "heldout_ids.npy", mmap_mode="r")
    heldout_dom = np.load(data_dir / "heldout_domain.npy")
    names = meta["domains"]

    total_tokens = int(args.max_tokens or t["total_tokens"])
    if total_tokens > len(train) * seq_len:
        print(f"[warn] corpus has {len(train) * seq_len / 1e6:.0f}M tokens < {total_tokens / 1e6:.0f}M"
              " requested; training on the whole corpus once")
        total_tokens = len(train) * seq_len
    every = int(args.checkpoint_every or t["checkpoint_every_tokens"])
    seqs_per_step = t["micro_batch"] * t["grad_accum"]
    tokens_per_step = seqs_per_step * seq_len
    total_steps = min(math.ceil(total_tokens / tokens_per_step), len(train) // seqs_per_step)
    assert total_steps * seqs_per_step <= len(train), \
        f"corpus has {len(train)} sequences, need {total_steps * seqs_per_step}"

    repo = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(repo)
    gen = load_generator(repo)
    disc, head = load_discriminator(repo)
    model = RTDModel(gen, disc, head).to(device)
    special_ids = torch.tensor(tok.all_special_ids, device=device)
    vocab = gen.config.vocab_size

    no_decay = ("bias", "LayerNorm.weight", "delta")
    params = [p for p in model.parameters() if p.requires_grad]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in named if not n.endswith(no_decay)],
               "weight_decay": t["weight_decay"]},
              {"params": [p for n, p in named if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=t["learning_rate"], betas=(0.9, 0.98), eps=1e-6)
    sched = get_linear_schedule_with_warmup(opt, int(t["warmup_ratio"] * total_steps), total_steps)

    step, tokens_seen, next_ckpt, history = 0, 0, every, []
    dom_tokens = np.zeros(len(names), dtype=np.int64)
    if args.resume:
        rp = Path(args.resume)
        st = torch.load(rp / "rtd_state.pt", map_location=device, weights_only=False)
        model.gen.load_state_dict(st["gen"])
        model.disc.load_state_dict(st["disc"])
        model.head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        prev = json.loads((rp / "pretrain_state.json").read_text())
        step, tokens_seen, history = prev["step"], prev["tokens_seen"], prev["history"]
        dom_tokens = np.array([prev["tokens_per_domain"][n] for n in names], dtype=np.int64)
        next_ckpt = (tokens_seen // every + 1) * every
        print(f"resumed at step {step}, {tokens_seen / 1e6:.0f}M tokens")

    rng = torch.Generator(device=device).manual_seed(t["seed"] + step)
    print(f"train seqs={len(train)} steps={total_steps} tokens/step={tokens_per_step} "
          f"total={total_tokens / 1e6:.0f}M ckpt every {every / 1e6:.0f}M")
    probe0 = probes(model, tok, device)
    if step == 0:
        print("probes @0:", json.dumps(probe0, indent=1))
        base_eval = evaluate(model, heldout, heldout_dom, names, special_ids, tok.mask_token_id,
                             vocab, device, t["rtd_lambda"])
        print("heldout @0:", json.dumps(base_eval, indent=1))
        history.append({"tokens": 0, "heldout": base_eval})

    model.train()
    t0, run = time.time(), {"loss": 0.0, "mlm": 0.0, "rtd": 0.0, "n": 0}
    while step < total_steps:
        base = step * seqs_per_step
        for a in range(t["grad_accum"]):
            sl = slice(base + a * t["micro_batch"], base + (a + 1) * t["micro_batch"])
            ids = torch.from_numpy(train[sl].astype(np.int64)).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, m, r, _ = rtd_step(model, ids, special_ids, tok.mask_token_id, vocab, rng,
                                         t["rtd_lambda"])
            (loss / t["grad_accum"]).backward()
            run["loss"] += loss.item(); run["mlm"] += m.item(); run["rtd"] += r.item(); run["n"] += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        dom_tokens += train_dom[base:base + seqs_per_step].sum(0)
        step += 1
        tokens_seen += tokens_per_step
        if step % t["log_every"] == 0:
            el = time.time() - t0
            n = run["n"]
            print(f"step {step}/{total_steps} tok {tokens_seen / 1e6:.1f}M loss {run['loss'] / n:.3f} "
                  f"mlm {run['mlm'] / n:.3f} rtd {run['rtd'] / n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{t['log_every'] * tokens_per_step / el / 1e3:.0f}k tok/s "
                  f"eta {(total_steps - step) * el / t['log_every'] / 60:.0f} min", flush=True)
            t0, run = time.time(), {"loss": 0.0, "mlm": 0.0, "rtd": 0.0, "n": 0}
        if tokens_seen >= next_ckpt or step == total_steps:
            ev = evaluate(model, heldout, heldout_dom, names, special_ids, tok.mask_token_id,
                          vocab, device, t["rtd_lambda"])
            pr = probes(model, tok, device)
            history.append({"tokens": tokens_seen, "step": step, "heldout": ev, "probes": pr})
            print(json.dumps(history[-1], indent=1))
            state = {"step": step, "tokens_seen": tokens_seen, "total_steps": total_steps,
                     "tokens_per_domain": {n: int(v) for n, v in zip(names, dom_tokens)},
                     "config": cfg, "history": history}
            save_checkpoint(out_dir / f"tokens-{round(tokens_seen / 1e6):03d}M", model, tok, opt,
                            sched, state)
            next_ckpt += every


if __name__ == "__main__":
    main()
