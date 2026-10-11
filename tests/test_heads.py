"""v5 structured heads (src/heads.py) and encoder-swap windowing (tokenize_bio.window_encode)."""
import itertools

import torch
from transformers import XLMRobertaConfig, XLMRobertaForTokenClassification

from heads import CRF, StructHeadTokenClassifier, is_struct_head_model, pack

LABELS = ["O", "B-PERSON", "I-PERSON", "B-ACCOUNT", "I-ACCOUNT"]


def tiny_base(seed=0):
    torch.manual_seed(seed)
    cfg = XLMRobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=64, num_labels=len(LABELS),
                           id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    return XLMRobertaForTokenClassification(cfg).eval()


def _random_crf(T=3, seed=0):
    torch.manual_seed(seed)
    crf = CRF(T)
    for p in crf.parameters():
        p.data.normal_()
    return crf


def _path_score(crf, em, path):
    s = crf.start[path[0]] + em[0, path[0]]
    for a, b, t in zip(path, path[1:], range(1, len(path))):
        s = s + crf.trans[a, b] + em[t, b]
    return s + crf.end[path[-1]]


def test_crf_nll_and_viterbi_match_brute_force():
    T, L = 3, 4
    crf = _random_crf(T)
    em = torch.randn(2, L, T)
    lengths = [4, 2]
    mask = torch.arange(L)[None] < torch.tensor(lengths)[:, None]
    tags = torch.tensor([[0, 2, 1, 1], [2, 0, 0, 0]])
    nll, best = [], crf.viterbi(em, mask)
    for r, n in enumerate(lengths):
        paths = list(itertools.product(range(T), repeat=n))
        scores = torch.stack([_path_score(crf, em[r], p) for p in paths])
        nll.append(torch.logsumexp(scores, 0) - _path_score(crf, em[r], tags[r, :n].tolist()))
        assert best[r, :n].tolist() == list(paths[int(scores.argmax())])
    assert torch.allclose(crf.nll(em, tags, mask), torch.stack(nll).mean(), atol=1e-4)


def test_pack_left_aligns_labelled_positions():
    m = torch.tensor([[False, True, False, True], [True, False, False, False]])
    order, pm = pack(m)
    assert order[0, :2].tolist() == [1, 3] and order[1, 0].tolist() == 0
    assert pm.tolist() == [[True, True], [True, False]]


def test_span_targets_from_bio_skip_ignored_tokens():
    model = StructHeadTokenClassifier(tiny_base(), "span")
    # B-PERSON, (-100 continuation), I-PERSON, O, B-ACCOUNT, B-ACCOUNT
    labels = torch.tensor([[-100, 1, -100, 2, 0, 3, 3, -100]])
    tgt = model._span_targets(labels)
    spans = sorted(tuple(x) for x in tgt[0].nonzero().tolist())
    assert spans == [(0, 1, 3), (1, 5, 5), (1, 6, 6)]          # (type, start, end)


def test_struct_heads_train_decode_and_round_trip(tmp_path):
    ids = torch.randint(5, 100, (2, 10))
    att = torch.ones_like(ids)
    labels = torch.tensor([[-100, 1, 2, 0, 3, 4, 4, 0, 0, -100]] * 2)
    for head in ("crf", "span"):
        model = StructHeadTokenClassifier(tiny_base(), head)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        first = None
        for _ in range(60):
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
            first = first if first is not None else float(loss)
            opt.zero_grad(); loss.backward(); opt.step()
        assert float(loss) < first * 0.5, head
        model.eval()
        valid = labels != -100
        lab, conf = model.decode(ids, att, valid)
        assert lab[valid].tolist() == labels[valid].tolist(), head
        assert (conf[valid] > 0).all()
        # Trainer eval path: one-hot decoded labels
        out = model(input_ids=ids, attention_mask=att, labels=labels)
        assert out.logits.argmax(-1)[valid].tolist() == labels[valid].tolist()
        model.save_pretrained(tmp_path / head)
        assert is_struct_head_model(tmp_path / head)
        again = StructHeadTokenClassifier.from_pretrained(tmp_path / head).eval()
        assert again.decode(ids, att, valid)[0].tolist() == lab.tolist()


def test_new_parameters_are_head_only():
    for head, n in (("crf", 3), ("span", 2)):
        m = StructHeadTokenClassifier(tiny_base(), head)
        assert len(m.new_parameters()) == n


class _Tok:
    """Minimal tokenizer stub: one token per character."""
    def __init__(self, cls, sep, bos, eos):
        self.cls_token_id, self.sep_token_id, self.bos_token_id, self.eos_token_id = cls, sep, bos, eos

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"input_ids": [10 + ord(c) % 50 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_window_encode_specials_for_encoders_without_cls():
    from tokenize_bio import window_encode
    text = "abcdefghij"
    for tok, pre, post in ((_Tok(0, 2, 0, 2), [0], [2]),            # XLM-R: <s> ... </s> (unchanged)
                           (_Tok(None, None, 7, 8), [7], [8]),      # EuroBERT: BOS ... EOS
                           (_Tok(None, None, None, 9), [], [9])):   # Qwen3: ... EOS only
        wins = window_encode(tok, text, max_length=6, stride=2, show_breaks=False)
        for w in wins:
            assert len(w["input_ids"]) <= 6
            assert w["input_ids"][:len(pre)] == pre and w["input_ids"][len(w["input_ids"]) - len(post):] == post
            assert sum(w["special_tokens_mask"]) == len(pre) + len(post)
        covered = {o for w in wins for o, sp in zip(w["offset_mapping"], w["special_tokens_mask"]) if not sp}
        assert covered == {(i, i + 1) for i in range(len(text))}


def test_decode_many_equals_decode_per_chunk():
    """The predictor's batched decode (one Viterbi over all windows / one transfer per chunk) must give
    exactly decode()'s labels and confidences."""
    torch.manual_seed(3)
    for head in ("crf", "span"):
        model = StructHeadTokenClassifier(tiny_base(), head).eval()
        if head == "crf":
            for p in model.crf.parameters():
                p.data.normal_()
        else:
            model.qk.weight.data.normal_(0, 1.0)
        chunks = []
        for B, T in ((3, 12), (2, 7), (1, 12)):          # different widths, as in a real document
            ids = torch.randint(5, 100, (B, T))
            valid = torch.rand(B, T) > 0.3
            valid[0, :] = False                           # a window without labelled tokens
            chunks.append((ids, torch.ones_like(ids), valid))
        for (ids, att, valid), (lab, conf) in zip(chunks, model.decode_many(chunks)):
            ol, oc = model.decode(ids, att, valid)
            assert ol.tolist() == lab, head
            assert torch.allclose(oc, torch.tensor(conf)), head


def test_boundary_smoothing_targets_word_space_and_mass():
    model = StructHeadTokenClassifier(tiny_base(), "span", span_loss="ce", bs_eps=0.2, bs_d=1)
    # words at token positions 1, 3, 4, 6 (2 and 5 are continuations); gold PERSON = words 3..4
    labels = torch.tensor([[-100, 0, -100, 1, 2, -100, 0, -100]])
    valid = labels != -100
    tgt = model._span_targets(labels)
    y = model._smoothed_targets(tgt, valid)
    assert abs(float(y.sum()) - 1.0) < 1e-6                       # mass conserved
    assert abs(float(y[0, 0, 3, 4]) - 0.8) < 1e-6                 # gold keeps 1 - eps
    nb = sorted((a, b) for a, b in (y[0, 0] > 0).nonzero().tolist() if (a, b) != (3, 4))
    # distance-1 neighbours in WORD positions: start 1 or 4 (end 4), end 3 or 6 (start 3); never position 2 / 5
    assert nb == [(1, 4), (3, 3), (3, 6), (4, 4)]
    assert torch.allclose(y[0, 0, 1, 4], torch.tensor(0.05))
    plain = StructHeadTokenClassifier(tiny_base(), "span", span_loss="ce")
    assert torch.equal(plain._smoothed_targets(tgt, valid), tgt.float())


def test_ce_span_loss_trains_decodes_and_round_trips(tmp_path):
    ids = torch.randint(5, 100, (2, 10))
    att = torch.ones_like(ids)
    labels = torch.tensor([[-100, 1, 2, 0, 3, 4, 4, 0, 0, -100]] * 2)
    for eps in (0.0, 0.1):
        model = StructHeadTokenClassifier(tiny_base(), "span", span_loss="ce", bs_eps=eps)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        first = None
        for _ in range(80):
            loss = model(input_ids=ids, attention_mask=att, labels=labels).loss
            first = first if first is not None else float(loss.detach())
            opt.zero_grad(); loss.backward(); opt.step()
        assert float(loss) < first * 0.5, eps
        model.eval()
        valid = labels != -100
        lab, _ = model.decode(ids, att, valid)
        assert lab[valid].tolist() == labels[valid].tolist(), eps
        model.save_pretrained(tmp_path / f"m{eps}")
        again = StructHeadTokenClassifier.from_pretrained(tmp_path / f"m{eps}")
        assert (again.span_loss, again.bs_eps, again.bs_d) == ("ce", eps, 1)
