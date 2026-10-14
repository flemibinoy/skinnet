from typing import Protocol

import pandas as pd


class DatasetAdapter(Protocol):
    """Future adapters must supply paths, labels, grouping and annotation provenance."""

    def load(self) -> pd.DataFrame: ...
