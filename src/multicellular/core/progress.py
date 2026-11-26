# core/progress.py

import tqdm as _tqdm


def _in_notebook() -> bool:
    """Whether we're running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def progress(iterable, disable=False, **kwargs):
    """
    A tqdm progress bar that behaves in notebooks.

    In a terminal this is plain `tqdm`. In Jupyter, plain tqdm's
    carriage-return refreshes each land as a separate output line, so
    instead this shows a single widget bar (`tqdm.notebook`) when
    ipywidgets is installed, and no bar at all otherwise.
    """
    if disable or not _in_notebook():
        return _tqdm.tqdm(iterable, disable=disable, **kwargs)
    try:
        import ipywidgets  # noqa: F401
        from tqdm.notebook import tqdm as notebook_tqdm

        return notebook_tqdm(iterable, **kwargs)
    except ImportError:
        return _tqdm.tqdm(iterable, disable=True, **kwargs)
