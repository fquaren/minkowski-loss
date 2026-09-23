"""Markdown tables without the optional `tabulate` dependency (not in dl-stable)."""

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = ".3g", index: bool = True) -> str:
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)
