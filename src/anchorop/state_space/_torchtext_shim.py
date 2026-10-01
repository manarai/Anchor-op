"""Lightweight ``torchtext`` shim so ``import scgpt`` works on macOS /
newer torch where the compiled ``libtorchtext.so`` has an ABI mismatch
against the installed torch runtime.

scGPT 0.2.4 only uses ``torchtext.vocab.Vocab`` and ``torchtext.vocab.vocab``
as a thin dictionary of gene-symbol → index. Both are trivially replaced
by a plain ``dict``. Importing this module installs the shim in
``sys.modules`` and must happen BEFORE ``import scgpt`` anywhere in the
process.

Usage::

    from anchorop.state_space import _torchtext_shim  # noqa: F401
    import scgpt

or, equivalently, call :func:`install` explicitly.
"""
from __future__ import annotations

import sys
import types


class _DictVocab:
    """Minimal `torchtext.vocab.Vocab` replacement — just a dict of
    gene-symbol → integer index."""

    def __init__(self, mapping=None):
        self._m = dict(mapping) if mapping else {}
        self._default = 0  # index returned for OOV tokens by default

    def __len__(self):
        return len(self._m)

    def __contains__(self, k):
        return k in self._m

    def __getitem__(self, k):
        return self._m.get(k, self._default)

    def __call__(self, keys):
        """Batch lookup: list of tokens → list of indices."""
        return [self._m.get(k, self._default) for k in keys]

    def get_itos(self):
        return list(self._m.keys())

    def get_stoi(self):
        return dict(self._m)

    def set_default_index(self, i):
        self._default = int(i)


def _vocab_fn(ordered_dict, specials=None, min_freq=1):
    """Replacement for ``torchtext.vocab.vocab``: build a `_DictVocab`
    from an OrderedDict-like mapping with optional specials in front."""
    v = _DictVocab()
    if specials:
        for i, t in enumerate(specials):
            v._m[t] = i
    offset = len(v._m)
    items = ordered_dict.items() if hasattr(ordered_dict, "items") else ordered_dict
    for i, pair in enumerate(items):
        if isinstance(pair, tuple):
            tok, _freq = pair
        else:
            tok = pair
        if tok not in v._m:
            v._m[tok] = offset
            offset += 1
    return v


def install():
    """Install the torchtext shim in ``sys.modules`` if not already
    installed. Safe to call multiple times."""
    if "torchtext" in sys.modules and getattr(
            sys.modules["torchtext"], "__anchorop_shim__", False):
        return  # already installed by this shim
    tt = types.ModuleType("torchtext")
    tt_vocab = types.ModuleType("torchtext.vocab")
    tt_vocab.Vocab = _DictVocab
    tt_vocab.vocab = _vocab_fn
    tt.vocab = tt_vocab
    tt.__anchorop_shim__ = True
    sys.modules["torchtext"] = tt
    sys.modules["torchtext.vocab"] = tt_vocab


# Install on import so the first use (`from anchorop.state_space._torchtext_shim
# import install; install()`) also works via side effect.
install()
