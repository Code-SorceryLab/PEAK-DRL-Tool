"""
Regression guard for the model=rppo NameError.

train.py references RecurrentEvalCallback at its recurrent-eval call site
(the branch taken when model_name in {'rppo','recurrent_ppo'}). If the import
is missing, that line raises NameError at runtime. Importing the trainer module
is safe: all training execution is gated behind @hydra.main + the
`if __name__ == "__main__"` guard, so import alone runs no training.
"""
import importlib


def test_train_module_imports_without_error():
    # Must not raise (e.g. ImportError) on import.
    importlib.import_module("code.scripts.train")


def test_recurrent_eval_callback_symbol_resolves_in_train_namespace():
    train = importlib.import_module("code.scripts.train")
    # This is the exact name referenced at the model=rppo call site (train.py:1166).
    # Before the fix this attribute does not exist -> the call site would NameError.
    assert hasattr(train, "RecurrentEvalCallback"), (
        "train.py references RecurrentEvalCallback (line ~1166) but does not "
        "import it; model=rppo would crash with NameError."
    )
    # And it must be the real callback class, not some unrelated rebinding.
    from code.callbacks.RecurrentEvalCallback import RecurrentEvalCallback
    assert train.RecurrentEvalCallback is RecurrentEvalCallback
