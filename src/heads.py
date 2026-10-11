"""Structured token-classification heads (v5 grid, user request 2026-10-10). Same encoder and data as
v5.0; only the head changes.

    crf   encoder -> dropout -> linear (the base model's classifier = emission scores) -> linear-chain CRF
          over the word-initial tokens (the tokens that carry a label). Trained by CRF log-likelihood,
          decoded by Viterbi. The transition scores get their own learning rate (train.py --head_lr):
          at the encoder's 2e-5 they would barely move from their zero start.
    span  encoder -> dropout -> GlobalPointer (Su et al. 2022): for every entity type a score for every
          (start word, end word) pair, from rotary-position query/key projections. Trained with the
          multi-label categorical cross-entropy over all pairs; decoded as the highest-scoring
          non-overlapping spans with score > 0.
          Loss options (span head): "gp" (default) GlobalPointer's multi-label categorical CE with 0/1 targets;
          "ce" per-span softmax CE over {not-an-entity, 8 types} with the not-an-entity logit fixed at 0 (the
          span-classification loss of Zhu & Li, ACL 2022), which accepts soft targets: with bs_eps > 0,
          BOUNDARY SMOOTHING moves eps of each gold span's mass to the same-type spans whose (start, end) is
          within Manhattan distance bs_d words of it. Decoding is unchanged (a type logit > 0 = beats O).

Both see only word-initial tokens (labels != -100 in training, word_starts() at inference), exactly
the tokens v5.0's softmax head is trained and scored on. In eval mode with labels (Trainer) and in
decode() (PIIPredictor) they return per-token BIO labels, so compute_metrics, checkpoint choice and
the predictor are unchanged.

Saved as: the base model (save_pretrained) + struct_head.pt (head weights) + struct_head.json.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

CONFIG_NAME, WEIGHTS_NAME = "struct_head.json", "struct_head.pt"
NEG = -1e12


def pack(mask: torch.Tensor):
    """Left-align the True positions of each row: (index (B, L), packed mask (B, L)), L = max count."""
    n = mask.sum(1)
    L = max(int(n.max()), 1)
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)[:, :L]
    return order, torch.arange(L, device=mask.device)[None] < n[:, None]


class CRF(nn.Module):
    """Linear-chain CRF over packed (left-aligned) sequences."""

    def __init__(self, n_tags: int):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))      # trans[i, j]: tag i -> tag j

    def nll(self, em, tags, mask):
        """Mean negative log-likelihood; em (B, L, T) float, tags (B, L), mask (B, L) prefix mask."""
        B, L, T = em.shape
        has = mask[:, 0]
        tags = tags.clamp(min=0)
        # gold path score
        score = self.start[tags[:, 0]] + em[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, L):
            m = mask[:, t]
            step = self.trans[tags[:, t - 1], tags[:, t]] + em[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * m
        last = tags.gather(1, (mask.sum(1) - 1).clamp(min=0)[:, None]).squeeze(1)
        score = score + self.end[last]
        # partition function
        alpha = self.start[None] + em[:, 0]
        for t in range(1, L):
            nxt = torch.logsumexp(alpha[:, :, None] + self.trans[None] + em[:, t, None, :], dim=1)
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        logz = torch.logsumexp(alpha + self.end[None], dim=1)
        return ((logz - score) * has).sum() / has.sum().clamp(min=1)

    @torch.no_grad()
    def viterbi(self, em, mask):
        B, L, T = em.shape
        score = self.start[None] + em[:, 0]
        back = []
        for t in range(1, L):
            s = score[:, :, None] + self.trans[None]
            best, idx = s.max(1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t, None], nxt, score)
            back.append(torch.where(mask[:, t, None], idx, torch.arange(T, device=em.device)[None].expand(B, T)))
        last = (score + self.end[None]).argmax(1)
        path = [last]
        for idx in reversed(back):
            last = idx.gather(1, last[:, None]).squeeze(1)
            path.append(last)
        return torch.stack(path[::-1], 1)                         # (B, L); padded positions repeat


def rope(x):
    """Rotary position embedding over dim -1 of x (B, T, H, D)."""
    T, D = x.shape[1], x.shape[-1]
    pos = torch.arange(T, device=x.device, dtype=torch.float32)[:, None]
    inv = 10000 ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    ang = pos * inv[None]                                         # (T, D/2)
    sin, cos = ang.sin()[None, :, None], ang.cos()[None, :, None]
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class StructHeadTokenClassifier(nn.Module):
    def __init__(self, base: nn.Module, head: str, span_dim: int = 64, max_span: int = 160,
                 span_loss: str = "gp", bs_eps: float = 0.0, bs_d: int = 1):
        super().__init__()
        assert head in ("crf", "span"), head
        assert span_loss in ("gp", "ce"), span_loss
        assert bs_eps == 0 or span_loss == "ce", "boundary smoothing needs span_loss='ce' (soft targets)"
        self.base, self.config, self.head = base, base.config, head
        self.span_dim, self.max_span = span_dim, max_span
        self.span_loss, self.bs_eps, self.bs_d = span_loss, bs_eps, bs_d
        l2i = base.config.label2id
        self.entities = sorted({k[2:] for k in l2i if k != "O"}, key=lambda e: l2i["B-" + e])
        self.o_id = l2i["O"]
        self.b_ids = [l2i["B-" + e] for e in self.entities]
        self.i_ids = [l2i["I-" + e] for e in self.entities]
        if head == "crf":
            self.crf = CRF(len(l2i))
        else:
            self.qk = nn.Linear(base.config.hidden_size, len(self.entities) * 2 * span_dim)

    @property
    def encoder(self):
        return getattr(self.base, self.base.base_model_prefix)

    def new_parameters(self):
        """Parameters that start from zero / random and get the head learning rate."""
        return list(self.crf.parameters()) if self.head == "crf" else list(self.qk.parameters())

    # ------------------------------------------------------------------ scores
    def _emissions(self, input_ids, attention_mask):
        return self.base(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    def _span_scores(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # the same features the base model's own classifier reads: XLM-R dropout; ModernBERT / mmBERT its
        # prediction head (dense + act + norm) then drop
        if isinstance(getattr(self.base, "head", None), nn.Module):
            h = self.base.head(h)
        drop = getattr(self.base, "dropout", None) or getattr(self.base, "drop", None)
        h = drop(h) if isinstance(drop, nn.Module) else h
        B, T, _ = h.shape
        qk = self.qk(h).view(B, T, len(self.entities), 2, self.span_dim)
        q, k = rope(qk[..., 0, :]), rope(qk[..., 1, :])
        return torch.einsum("bmed,bned->bemn", q, k) / math.sqrt(self.span_dim)     # (B, E, T, T)

    def _pair_mask(self, valid):
        T = valid.shape[1]
        i = torch.arange(T, device=valid.device)
        band = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < self.max_span)
        return valid[:, None, :, None] & valid[:, None, None, :] & band[None, None]

    def _span_targets(self, labels):
        """(B, E, T, T) gold span matrix from BIO labels (-100 positions are skipped)."""
        B, T = labels.shape
        tgt = torch.zeros(B, len(self.entities), T, T, dtype=torch.bool, device=labels.device)
        b_of = {b: e for e, b in enumerate(self.b_ids)}
        i_of = {i: e for e, i in enumerate(self.i_ids)}
        for r, row in enumerate(labels.tolist()):
            cur = None                                            # (type, start, last)
            for t, l in enumerate(row):
                if l == -100:
                    continue
                if l in i_of and cur and cur[0] == i_of[l]:
                    cur = (cur[0], cur[1], t)
                    continue
                if cur:
                    tgt[r, cur[0], cur[1], cur[2]] = True
                cur = (b_of[l], t, t) if l in b_of else None
            if cur:
                tgt[r, cur[0], cur[1], cur[2]] = True
        return tgt

    def _smoothed_targets(self, tgt, valid):
        """Boundary smoothing (Zhu & Li 2022) on the (B, E, T, T) gold matrix: each gold span keeps 1 - eps,
        eps is shared equally by the same-type spans whose start / end differ by at most bs_d WORDS in total
        (word = labelled token; spans past the width limit or reversed are not candidates)."""
        y = tgt.float()
        if not self.bs_eps:
            return y
        eps, D = self.bs_eps, self.bs_d
        for r, row in enumerate(valid.tolist()):
            vp = [t for t, v in enumerate(row) if v]
            rank = {t: i for i, t in enumerate(vp)}
            for e, a, b in (tgt[r].nonzero().tolist()):
                ra, rb = rank[a], rank[b]
                nb = []
                for da in range(-D, D + 1):
                    for db in range(-D, D + 1):
                        if (da or db) and abs(da) + abs(db) <= D:
                            i, j = ra + da, rb + db
                            if 0 <= i <= j < len(vp) and vp[j] - vp[i] < self.max_span:
                                nb.append((vp[i], vp[j]))
                if not nb:
                    continue
                y[r, e, a, b] -= eps
                for i, j in nb:
                    y[r, e, i, j] += eps / len(nb)
        return y

    def _ce_loss(self, s, y, valid):
        """Per-span softmax CE over {O, types} with the O logit fixed at 0; soft targets y (B, E, T, T),
        O gets the remaining mass. Summed over the window's valid spans, averaged over windows."""
        pair = self._pair_mask(valid)[:, 0]                         # (B, T, T)
        logits = torch.cat([torch.zeros_like(s[:, :1]), s], 1)       # (B, 1 + E, T, T)
        logp = torch.log_softmax(logits.masked_fill(~pair[:, None], 0.0), 1)
        y_o = (1 - y.sum(1, keepdim=True)).clamp(min=0)
        tgt = torch.cat([y_o, y], 1)
        tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-9)          # overlapping smoothing mass: renormalise
        return (-(tgt * logp).sum(1) * pair).sum() / s.shape[0]

    # ------------------------------------------------------------------ decoding
    @torch.no_grad()
    def decode(self, input_ids, attention_mask, valid):
        """(labels (B, T) long, confidence (B, T) float) at `valid` tokens; other tokens get O / 0."""
        B, T = input_ids.shape
        out = torch.full((B, T), self.o_id, dtype=torch.long, device=input_ids.device)
        conf = torch.zeros(B, T, device=input_ids.device)
        if self.head == "crf":
            em = self._emissions(input_ids, attention_mask)
            order, pm = pack(valid)
            pem = em.gather(1, order[:, :, None].expand(-1, -1, em.shape[-1]))
            path = self.crf.viterbi(pem, pm)
            p = pem.softmax(-1).gather(2, path[:, :, None]).squeeze(2)
            for r in range(B):
                n = int(pm[r].sum())
                out[r, order[r, :n]] = path[r, :n]
                conf[r, order[r, :n]] = p[r, :n]
            return out, conf
        s = self._span_scores(input_ids, attention_mask).float()
        s = s.masked_fill(~self._pair_mask(valid), NEG)
        conf[valid] = 0.5
        for r in range(B):
            cand = (s[r] > 0).nonzero().tolist()
            cand.sort(key=lambda c: -float(s[r, c[0], c[1], c[2]]))
            taken = torch.zeros(T, dtype=torch.bool)
            for e, a, b in cand:
                if taken[a:b + 1].any():
                    continue
                taken[a:b + 1] = True
                p = float(torch.sigmoid(s[r, e, a, b]))
                pos = [t for t in range(a, b + 1) if valid[r, t]]
                out[r, pos[0]] = self.b_ids[e]
                out[r, pos[1:]] = self.i_ids[e]
                conf[r, pos] = p
        return out, conf

    @torch.no_grad()
    def span_scores_masked(self, input_ids, attention_mask, valid):
        """Span head: (B, E, T, T) logits, invalid pairs at NEG (predictor merge=avg)."""
        return self._span_scores(input_ids, attention_mask).float().masked_fill(~self._pair_mask(valid), NEG)

    @torch.no_grad()
    def decode_many(self, chunks):
        """Batched decode() for the predictor: chunks = [(input_ids, attention_mask, valid), ...] (one
        document's window batches). Same output as decode() per chunk, as CPU lists [(labels, conf)], but
        CRF runs ONE Viterbi over all windows (the step loop is per position, not per window) and the
        span decoder moves its candidates to the CPU in one transfer per chunk instead of one per span."""
        res = []
        if self.head == "crf":
            packed = []
            for ids, att, valid in chunks:
                em = self._emissions(ids, att)
                order, pm = pack(valid)
                packed.append((em.gather(1, order[:, :, None].expand(-1, -1, em.shape[-1])), pm, order, ids.shape))
            L = max(p[0].shape[1] for p in packed)
            pad = lambda x, v: nn.functional.pad(x, (0, 0, 0, L - x.shape[1]) if x.dim() == 3 else (0, L - x.shape[1]), value=v)
            pem = torch.cat([pad(p[0], 0.0) for p in packed])
            pm = torch.cat([pad(p[1], False) for p in packed])
            path = self.crf.viterbi(pem, pm)
            conf = pem.softmax(-1).gather(2, path[:, :, None]).squeeze(2)
            r0 = 0
            for (_, m, order, shape) in packed:
                B, n = shape[0], m.shape[1]
                lab = torch.full(shape, self.o_id, dtype=torch.long, device=path.device)
                cf = torch.zeros(shape, device=path.device)
                pth, c, mm = path[r0:r0 + B, :n], conf[r0:r0 + B, :n], m
                # positions past a row's count are its non-word-initial tokens: they keep O / 0
                lab.scatter_(1, order, torch.where(mm, pth, lab.gather(1, order)))
                cf.scatter_(1, order, torch.where(mm, c, cf.gather(1, order)))
                res.append((lab.tolist(), cf.tolist()))
                r0 += B
            return res
        for ids, att, valid in chunks:
            B, T = ids.shape
            s = self._span_scores(ids, att).float().masked_fill(~self._pair_mask(valid), NEG)
            pos = s > 0
            cand = pos.nonzero().tolist()                         # (row, type, start, end), row-major
            sc = s[pos].tolist()
            pr = torch.sigmoid(s[pos]).tolist()
            vl = valid.tolist()
            lab = [[self.o_id] * T for _ in range(B)]
            cf = [[0.5 if v else 0.0 for v in row] for row in vl]
            by_row = [[] for _ in range(B)]
            for (r, e, a, b), x, p in zip(cand, sc, pr):
                by_row[r].append((x, e, a, b, p))
            for r, cs in enumerate(by_row):
                cs.sort(key=lambda c: -c[0])                      # stable: ties keep (type, start, end) order
                taken = [False] * T
                for x, e, a, b, p in cs:
                    if any(taken[a:b + 1]):
                        continue
                    for t in range(a, b + 1):
                        taken[t] = True
                    pos_ = [t for t in range(a, b + 1) if vl[r][t]]
                    lab[r][pos_[0]] = self.b_ids[e]
                    for t in pos_[1:]:
                        lab[r][t] = self.i_ids[e]
                    for t in pos_:
                        cf[r][t] = p
            res.append((lab, cf))
        return res

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        if labels is None or not self.training:
            if labels is None:
                return TokenClassifierOutput(logits=self._emissions(input_ids, attention_mask)
                                             if self.head == "crf" else None)
            # Trainer eval: decoded labels as one-hot logits, so seqeval scores the structured decode
            lab, _ = self.decode(input_ids, attention_mask, labels != -100)
            logits = nn.functional.one_hot(lab, len(self.config.label2id)).float()
            return TokenClassifierOutput(loss=torch.zeros((), device=lab.device), logits=logits)
        valid = labels != -100
        if self.head == "crf":
            em = self._emissions(input_ids, attention_mask)
            order, pm = pack(valid)
            pem = em.gather(1, order[:, :, None].expand(-1, -1, em.shape[-1]))
            return TokenClassifierOutput(loss=self.crf.nll(pem, labels.gather(1, order), pm), logits=em)
        s = self._span_scores(input_ids, attention_mask).float()
        if self.span_loss == "ce":
            y = self._smoothed_targets(self._span_targets(labels), valid)
            return TokenClassifierOutput(loss=self._ce_loss(s, y, valid), logits=None)
        y = self._span_targets(labels).float()
        s = s.masked_fill(~self._pair_mask(valid), NEG)
        # multi-label categorical cross-entropy (GlobalPointer), per (window, type)
        s = (1 - 2 * y) * s
        neg = (s - y * 1e12).flatten(2)
        pos = (s - (1 - y) * 1e12).flatten(2)
        zero = torch.zeros_like(neg[..., :1])
        loss = (torch.logsumexp(torch.cat([neg, zero], -1), -1) + torch.logsumexp(torch.cat([pos, zero], -1), -1)).mean()
        return TokenClassifierOutput(loss=loss, logits=None)

    # ------------------------------------------------------------------ save / load
    def save_pretrained(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(out)
        head = self.crf if self.head == "crf" else self.qk
        torch.save(head.state_dict(), out / WEIGHTS_NAME)
        (out / CONFIG_NAME).write_text(json.dumps({"head": self.head, "span_dim": self.span_dim,
                                                   "max_span": self.max_span, "span_loss": self.span_loss,
                                                   "bs_eps": self.bs_eps, "bs_d": self.bs_d}), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir, dtype=torch.float32):
        d = Path(model_dir)
        cfg = json.loads((d / CONFIG_NAME).read_text(encoding="utf-8"))
        base = AutoModelForTokenClassification.from_pretrained(str(d), dtype=dtype)
        model = cls(base, cfg["head"], cfg.get("span_dim", 64), cfg.get("max_span", 160),
                    cfg.get("span_loss", "gp"), cfg.get("bs_eps", 0.0), cfg.get("bs_d", 1))
        head = model.crf if model.head == "crf" else model.qk
        head.load_state_dict(torch.load(d / WEIGHTS_NAME, map_location="cpu"))
        return model


def is_struct_head_model(model_dir) -> bool:
    return (Path(model_dir) / CONFIG_NAME).exists()
