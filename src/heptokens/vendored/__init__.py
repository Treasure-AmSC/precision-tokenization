"""Snapshot of the heptokens-new code used for the calorimeter-cell tokenizer and the jet-pT regression.

models/, data/base.py, data/collation.py, callbacks/recon.py: heptokens @ 73cb6ae (branch cells-paper-snapshot)
cocoa/: heptokens_cocoa (heptokens-cocoa @ cc20cbc, branch cells-paper-snapshot)
Only imports were changed. See README.md, "Reproducing the paper".
"""
import contextlib
import importlib
import sys

_B_MODULES = {
    "heptokens.models.vq_vae": "heptokens.vendored.models.vq_vae",
    "heptokens.models.coders": "heptokens.vendored.models.coders",
    "heptokens.models.transformer": "heptokens.vendored.models.transformer",
    "heptokens.models.utils": "heptokens.vendored.models.utils",
    "heptokens.models.vq_layers": "heptokens.vendored.models.vq_layers",
}


@contextlib.contextmanager
def b_module_names():
    """While unpickling a checkpoint, resolve heptokens.models.* to the vendored modules.

    Checkpoints pickle their encoder/decoder partials by module path. In the environment that produced the
    paper's cell and regression results, heptokens.models.* was this vendored code; in this repository it is
    the original heptokens package, which has no CellDecoder.
    """
    saved = {name: sys.modules.get(name) for name in _B_MODULES}
    try:
        for name, target in _B_MODULES.items():
            sys.modules[name] = importlib.import_module(target)
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
