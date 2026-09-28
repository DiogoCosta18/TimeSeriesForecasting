from __future__ import annotations

import pandas as pd

from .stl_transform import STLTransform


def make_component_targets(train_df: pd.DataFrame, season_length: int) -> tuple[pd.DataFrame, STLTransform]:
    tr = STLTransform(season_length).fit(train_df["y"].to_numpy(dtype=float))
    comps = tr.components()
    out = train_df.copy()
    out["trend"] = comps["trend"]
    out["seasonal"] = comps["seasonal"]
    out["residual"] = comps["residual"]
    out["nonseasonal"] = comps["trend"] + comps["residual"]
    return out, tr

