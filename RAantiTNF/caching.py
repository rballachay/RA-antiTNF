import os
from functools import wraps
from pathlib import Path

import pandas as pd


def cachewrapper(path, prefix):
    """caching decorator to save intermediate dfs,
    makes repeated testing much faster
    """
    if isinstance(prefix, tuple):
        fullpath = []
        for _pre in prefix:
            _pathi = str(Path(path) / f"{_pre}_cache.csv")
            fullpath.append(_pathi)
    else:
        fullpath = str(Path(path) / f"{prefix}_cache.csv")

    def decorator(function):
        if not isinstance(fullpath, list):

            @wraps(function)
            def wrapper(*args, **kwargs):
                if os.path.exists(fullpath):
                    df = pd.read_csv(fullpath)
                else:
                    df = function(*args, **kwargs)
                    df.to_csv(fullpath, index=False)
                return df

            return wrapper
        else:

            @wraps(function)
            def wrapper(*args, **kwargs):
                if all([os.path.exists(i) for i in fullpath]):
                    df = [pd.read_csv(i) for i in fullpath]
                else:
                    df = function(*args, **kwargs)
                    for _dfi, _pathi in zip(df, fullpath):
                        _dfi.to_csv(_pathi, index=False)
                return df

            return wrapper

    return decorator
