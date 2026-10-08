"""Language adapters (src/adapters.py) on a tiny random XLM-R."""
import torch
from transformers import XLMRobertaConfig, XLMRobertaForTokenClassification

from adapters import ADAPTERS, LANG_IDS, AdapterTokenClassifier


def tiny_base(seed=0):
    torch.manual_seed(seed)
    cfg = XLMRobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=64, num_labels=5)
    return XLMRobertaForTokenClassification(cfg).eval()


def batch(n=4, length=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, 100, (n, length), generator=g), torch.ones(n, length, dtype=torch.long)


def test_fresh_adapters_are_identity():
    base = tiny_base()
    ids, mask = batch()
    with torch.no_grad():
        ref = base(input_ids=ids, attention_mask=mask).logits
        model = AdapterTokenClassifier(tiny_base(), bottleneck=8).eval()
        got = model(input_ids=ids, attention_mask=mask, lang_ids=torch.tensor([0, 1, 2, 3])).logits
    assert torch.allclose(ref, got, atol=1e-6)


def test_mixed_batch_equals_single_language_batches():
    model = AdapterTokenClassifier(tiny_base(), bottleneck=8).eval()
    with torch.no_grad():
        for p in model.parameters():          # make adapters non-trivial and language-specific
            p.add_(torch.randn_like(p) * 0.05)
    ids, mask = batch()
    langs = torch.tensor([LANG_IDS["th"], LANG_IDS["ar"], LANG_IDS["th"], LANG_IDS["zh-Hant"]])
    with torch.no_grad():
        mixed = model(input_ids=ids, attention_mask=mask, lang_ids=langs).logits
        for i in range(len(ids)):
            single = model(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1], lang_ids=langs[i:i + 1]).logits
            assert torch.allclose(mixed[i], single[0], atol=1e-5), i
        other = model(input_ids=ids[:1], attention_mask=mask[:1], lang_ids=torch.tensor([LANG_IDS["en"]])).logits
    assert not torch.allclose(mixed[0], other[0])        # a different adapter changes the output


def test_only_adapters_and_head_train():
    model = AdapterTokenClassifier(tiny_base(), bottleneck=8)
    train = {n for n, p in model.named_parameters() if p.requires_grad}
    assert train and all(".adapters." in n or n.startswith("base.classifier.") for n in train)
    assert any(n.startswith("base.classifier.") for n in train)
    per = model.adapter_params()
    assert set(per) == set(ADAPTERS) and len(set(per.values())) == 1
    # 2 layers x (LN 2*32 + down 32*8+8 + up 8*32+32)
    assert per["th"] == 2 * (64 + 264 + 288)


def test_zh_variants_share_an_adapter():
    assert LANG_IDS["zh-Hans"] == LANG_IDS["zh-Hant"]
    assert len(set(LANG_IDS.values())) == 11


def test_save_and_load_round_trip(tmp_path):
    model = AdapterTokenClassifier(tiny_base(), bottleneck=8).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)
    model.save_pretrained(tmp_path)              # no base_dir: encoder saved alongside
    loaded = AdapterTokenClassifier.from_pretrained(tmp_path).eval()
    ids, mask = batch()
    langs = torch.tensor([0, 3, 5, 7])
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask, lang_ids=langs).logits
        b = loaded(input_ids=ids, attention_mask=mask, lang_ids=langs).logits
    assert torch.allclose(a, b, atol=1e-6)


def _identity_and_routing(base_factory):
    ids, mask = batch()
    with torch.no_grad():
        ref = base_factory().eval()(input_ids=ids, attention_mask=mask).logits
        model = AdapterTokenClassifier(base_factory(), bottleneck=8).eval()
        got = model(input_ids=ids, attention_mask=mask, lang_ids=torch.tensor([0, 1, 2, 3])).logits
        assert torch.allclose(ref, got, atol=1e-6)               # fresh adapters = identity
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        langs = torch.tensor([LANG_IDS["th"], LANG_IDS["ar"], LANG_IDS["th"], LANG_IDS["ko"]])
        mixed = model(input_ids=ids, attention_mask=mask, lang_ids=langs).logits
        one = model(input_ids=ids[1:2], attention_mask=mask[1:2], lang_ids=langs[1:2]).logits
        assert torch.allclose(mixed[1], one[0], atol=1e-5)
    assert all(".adapters." in n or n.startswith("base.classifier.")
               for n, p in model.named_parameters() if p.requires_grad)


def test_encoder_is_swappable_deberta_v2():
    """mDeBERTa-v3 (DeBERTa-v2 architecture) works as the shared encoder."""
    from transformers import DebertaV2Config, DebertaV2ForTokenClassification

    def make():
        torch.manual_seed(0)
        return DebertaV2ForTokenClassification(DebertaV2Config(
            vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
            max_position_embeddings=64, num_labels=5, relative_attention=True, position_buckets=16,
            pos_att_type=["p2c", "c2p"]))
    _identity_and_routing(make)


def test_encoder_is_swappable_bert():
    """BERT-architecture encoders (e.g. multilingual MiniLM / mBERT) work as the shared encoder."""
    from transformers import BertConfig, BertForTokenClassification

    def make():
        torch.manual_seed(0)
        return BertForTokenClassification(BertConfig(
            vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
            max_position_embeddings=64, num_labels=5))
    _identity_and_routing(make)


# ---------------------------------------------------------------- v5 grid: shared adapter (S1), freeze modes (H, P)
def test_shared_adapter_ignores_language_and_has_one_set():
    model = AdapterTokenClassifier(tiny_base(), bottleneck=8, shared=True).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ids, mask = batch()
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask, lang_ids=torch.tensor([0, 1, 2, 3])).logits
        b = model(input_ids=ids, attention_mask=mask, lang_ids=torch.tensor([5, 5, 5, 5])).logits
        c = model(input_ids=ids, attention_mask=mask).logits          # no routing needed at all
    assert torch.allclose(a, b) and torch.allclose(a, c)
    per = model.adapter_params()
    assert set(per) == {"shared"} and per["shared"] == 2 * (64 + 264 + 288)   # one set per layer


def test_shared_adapter_save_load_roundtrip(tmp_path):
    model = AdapterTokenClassifier(tiny_base(), bottleneck=8, shared=True).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".adapters." in n:
                p.add_(torch.randn_like(p) * 0.05)
    model.base.save_pretrained(tmp_path / "basecopy")
    model.save_pretrained(tmp_path / "m", base_dir=str(tmp_path / "basecopy"))
    loaded = AdapterTokenClassifier.from_pretrained(tmp_path / "m").eval()
    assert loaded.shared
    ids, mask = batch()
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids, attention_mask=mask).logits,
                              loaded(input_ids=ids, attention_mask=mask).logits, atol=1e-6)


def test_default_adapter_model_unchanged():
    """B1 / B' (queued) use the default: per-language adapters, routing required."""
    model = AdapterTokenClassifier(tiny_base(), bottleneck=8)
    assert not model.shared and set(model.adapter_params()) == set(ADAPTERS)
    import pytest
    ids, mask = batch()
    with pytest.raises(RuntimeError):
        model(input_ids=ids, attention_mask=mask)


def test_freeze_modes():
    from train import apply_freeze
    m = tiny_base()
    apply_freeze(m, {"mode": "head_only"})
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {"classifier.weight", "classifier.bias"}
    m = tiny_base()
    apply_freeze(m, {"mode": "top_k", "k": 1})
    train = {n for n, p in m.named_parameters() if p.requires_grad}
    assert {"classifier.weight", "classifier.bias"} <= train
    assert any(".encoder.layer.1." in n for n in train) and not any(".encoder.layer.0." in n for n in train)
    assert not any("embeddings" in n for n in train)
