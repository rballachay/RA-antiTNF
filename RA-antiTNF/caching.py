import os
from functools import wraps
from pathlib import Path

import pandas as pd


def cachewrapper(path, prefix):
    """caching decorator to save intermediate dfs,
    makes repeated testing much faster
    """
    path = str(Path(path) / f"{prefix}_cache.csv")

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = function(*args, **kwargs)
                df.to_csv(path, index=False)
            return df

        return wrapper

    return decorator
