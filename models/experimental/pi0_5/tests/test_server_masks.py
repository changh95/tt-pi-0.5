# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device): every server path hands the request's OWN language mask to ``sample_actions_fused``.

Before 2026-09-30 the batcher (which serves every request, batch 1 included) dropped the mask and the model fell back
to ``tokens != 0``. For tokenizer prompts the two agree, but a pre-tokenised request may carry id 0 inside the prompt,
and the fallback is not the contract. Needs fastapi / pydantic / PIL (the server's imports); skipped without them."""
import threading

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")
pytest.importorskip("PIL")

from models.experimental.pi0_5.server import app as A  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.calls = []
        self._default_noise_torch = torch.zeros(1, A.ACTION_HORIZON, A.ACTION_DIM)
        self.lock = threading.Lock()

    def sample_actions_fused(self, images, lang_tokens, noise=None, lang_masks=None):
        with self.lock:
            self.calls.append(dict(tokens=lang_tokens.clone(), masks=None if lang_masks is None else lang_masks.clone(),
                                   n_images=len(images)))
        b = lang_tokens.shape[0]
        # action row = the request's real-token count, so each future can be matched to its request
        n = (lang_masks if lang_masks is not None else lang_tokens != 0).long().sum(dim=1).float()
        return n.reshape(b, 1, 1).expand(b, A.ACTION_HORIZON, A.ACTION_DIM).contiguous()


def _request(tokens, token_len=8):
    ids, mask, n = A.tokens_from_request(tokens, token_len)
    return [A.black_image(), A.black_image()], ids, mask, n


def test_tokens_from_request_mask_is_the_length_not_nonzero():
    ids, mask, n = A.tokens_from_request([5, 0, 7], 8)
    assert n == 3 and mask[0].tolist() == [True] * 3 + [False] * 5
    assert not torch.equal(mask, ids != 0)  # the case the old fallback got wrong


def test_batcher_passes_the_request_mask_batch1():
    m = _FakeModel()
    b = A._Batcher(m, (1,), window_s=0.0)
    images, ids, mask, n = _request([5, 0, 7])
    actions, batched_as = b.submit(images, ids, mask, None).result(timeout=10)
    assert batched_as == 1 and float(actions[0, 0, 0]) == n == 3
    assert len(m.calls) == 1 and m.calls[0]["masks"] is not None
    assert torch.equal(m.calls[0]["masks"], mask)
    assert torch.equal(m.calls[0]["tokens"], ids)


def test_batcher_passes_every_mask_when_batching_and_padding():
    m = _FakeModel()
    b = A._Batcher(m, (1, 4), window_s=0.5)
    reqs = [_request(t) for t in ([5, 0, 7], [9], [3, 3, 0, 0, 3])]
    futs = [b.submit(im, ids, mask, None) for (im, ids, mask, _n) in reqs]
    outs = [f.result(timeout=10) for f in futs]
    for (acts, _bs), (_im, _ids, _mask, n) in zip(outs, reqs):
        assert float(acts[0, 0, 0]) == n
    masks = torch.cat([c["masks"] for c in m.calls], dim=0)
    tokens = torch.cat([c["tokens"] for c in m.calls], dim=0)
    assert all(c["masks"] is not None for c in m.calls)
    for (_im, ids, mask, _n) in reqs:  # every request's own mask reached the model next to its own tokens
        rows = [i for i in range(tokens.shape[0]) if torch.equal(tokens[i : i + 1], ids)]
        assert rows and all(torch.equal(masks[i : i + 1], mask) for i in rows)


def test_batcher_refuses_a_missing_mask():
    b = A._Batcher(_FakeModel(), (1,), window_s=0.0)
    images, ids, _mask, _n = _request([5])
    with pytest.raises(ValueError, match="lang mask"):
        b.submit(images, ids, None, None)


def test_dp_router_passes_the_mask():
    ms = [_FakeModel(), _FakeModel()]
    r = A._DPRouter([A._Batcher(m, (1,), window_s=0.0) for m in ms], [[0, 1], [2, 3]])
    images, ids, mask, n = _request([5, 0, 7])
    actions, _ = r.submit(images, ids, mask, None).result(timeout=10)
    assert float(actions[0, 0, 0]) == n
    calls = [c for m in ms for c in m.calls]
    assert len(calls) == 1 and torch.equal(calls[0]["masks"], mask)
