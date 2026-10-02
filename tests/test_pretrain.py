"""RTD pretraining pieces: released weights load, GDES starts from the released model,
masking/labels are right, packing keeps sequences intact, exported checkpoints fine-tune."""
import numpy as np
import pytest
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, DebertaV2Model

from build_pretrain_corpus import clean_email, pack
from pretrain_rtd import RTDModel, load_discriminator, load_generator, mask_tokens, rtd_step

REPO = "microsoft/deberta-v3-xsmall"


@pytest.fixture(scope="module")
def parts():
    tok = AutoTokenizer.from_pretrained(REPO)
    gen = load_generator(REPO)
    disc, head = load_discriminator(REPO)
    released = disc.embeddings.word_embeddings.weight.detach().clone()
    model = RTDModel(gen, disc, head).eval()
    return tok, model, released


def test_gdes_starts_at_released_embeddings(parts):
    _, model, released = parts
    assert torch.allclose(model.disc.embeddings.word_embeddings.weight, released, atol=1e-6)
    # the disc path must not push gradients into the generator's embedding
    ids = torch.tensor([[1, 100, 200, 2]])
    model.disc_logits(ids, torch.ones_like(ids)).sum().backward()
    assert model.gen.deberta.embeddings.word_embeddings.weight.grad is None
    assert model.disc.embeddings.word_embeddings.delta.grad is not None
    model.zero_grad(set_to_none=True)


def test_released_heads_work(parts):
    tok, model, _ = parts
    enc = tok("Your account [MASK] is 123456789.", return_tensors="pt")
    i = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0].item()
    with torch.no_grad():
        top = model.gen(**enc).logits[0, i].topk(5).indices
    assert "▁number" in tok.convert_ids_to_tokens(top)
    # released RTD head: a corrupted token scores higher than the clean ones around it
    clean = tok("The bank charged a monthly maintenance fee on the savings account.",
                return_tensors="pt").input_ids
    bad = clean.clone()
    j = 6
    bad[0, j] = tok.convert_tokens_to_ids("▁banana")
    with torch.no_grad():
        logits = model.disc_logits(bad, torch.ones_like(bad))[0]
    assert logits[j] == logits[1:-1].max()


def test_masking_and_labels(parts):
    tok, model, _ = parts
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 5000, (8, 128))
    ids[:, 0], ids[:, -1] = tok.cls_token_id, tok.sep_token_id
    special = torch.isin(ids, torch.tensor(tok.all_special_ids))
    inp, labels = mask_tokens(ids, special, 128000, tok.mask_token_id, g)
    sel = labels != -100
    assert not sel[special].any()
    assert 0.10 < sel.float().mean() < 0.20
    assert ((inp == tok.mask_token_id) & sel).sum() > 0.6 * sel.sum()
    assert torch.equal(inp[~sel], ids[~sel])
    loss, mlm, rtd, stats = rtd_step(model, ids, torch.tensor(tok.all_special_ids),
                                     tok.mask_token_id, 128000, g, 10)
    assert torch.isfinite(loss) and stats["replaced"] <= stats["masked"]


def test_export_loads_for_token_classification(parts, tmp_path):
    tok, model, released = parts
    model.export_discriminator().save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    m = AutoModelForTokenClassification.from_pretrained(tmp_path, num_labels=17)
    assert torch.allclose(m.deberta.embeddings.word_embeddings.weight, released, atol=1e-6)
    ref = DebertaV2Model.from_pretrained(REPO, dtype=torch.float32)
    k = "encoder.layer.11.output.dense.weight"
    assert torch.allclose(m.deberta.state_dict()[k].float(), ref.state_dict()[k])


def test_released_embedding_is_gdes_sum():
    """The released discriminator embedding = generator embedding + GDES delta (`_weight`),
    i.e. the checkpoint was trained with the sharing pretrain_rtd.py reproduces."""
    from huggingface_hub import hf_hub_download
    d = torch.load(hf_hub_download(REPO, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    g = torch.load(hf_hub_download(REPO, "pytorch_model.generator.bin"), map_location="cpu",
                   weights_only=True)
    key = "deberta.embeddings.word_embeddings."
    total = g[key + "weight"].float() + d[key + "_weight"].float()
    assert (d[key + "weight"].float() - total).abs().max() < 5e-3   # fp16 rounding


def test_pack_keeps_tokens_and_domains():
    docs = [(0, np.arange(10, 1010, dtype=np.int32)), (1, np.arange(5000, 5300, dtype=np.int32))]
    seqs, doms = pack(docs, 64, 1, 2, 2)
    assert seqs.shape[1] == 64 and (seqs[:, 0] == 1).all() and (seqs[:, -1] == 2).all()
    body = seqs[:, 1:-1].ravel()
    assert body[:1000].tolist() == list(range(10, 1010))
    assert doms.sum() == seqs.shape[0] * 62
    assert doms[0, 0] == 62 and doms[0, 1] == 0


def test_clean_email_drops_quoted_text():
    body = ("Hi John,\nPlease see the attached statement.\nThanks\n"
            "-----Original Message-----\nFrom: Jane\nOld text")
    assert clean_email(body).strip().endswith("Thanks")
