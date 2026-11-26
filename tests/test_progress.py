import builtins

from multicellular.core import progress as progress_module
from multicellular.core.progress import progress


def test_terminal_bar_enabled(monkeypatch):
    monkeypatch.setattr(progress_module, "_in_notebook", lambda: False)
    bar = progress(range(3))
    assert not bar.disable
    assert list(bar) == [0, 1, 2]


def test_notebook_without_ipywidgets_is_silent(monkeypatch):
    monkeypatch.setattr(progress_module, "_in_notebook", lambda: True)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ipywidgets":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bar = progress(range(3))
    assert bar.disable
    assert list(bar) == [0, 1, 2]


def test_disable_respected(monkeypatch):
    monkeypatch.setattr(progress_module, "_in_notebook", lambda: False)
    assert progress(range(3), disable=True).disable
