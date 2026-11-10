"""
Cell-by-cell diff of the Step 2-4 tables in two results folders.

Rows are matched on their key columns (county FIPS, factor, variant, ...).
Numeric cells differ if not equal to rtol 1e-9; second_sig_fig_changed says
whether the value rounded to 2 significant figures changed.

Run from the repo root:
    venv/bin/python analysis/compare_results.py analysis/results analysis/results_v2 \
        analysis/results_v2/v1_vs_v2_differences.csv
"""
import os
import sys

import numpy as np
import pandas as pd

# key columns per table
KEYS = {"step2_reff_main.csv": ["countyFIPS"], "step2_reff_wave_level.csv": ["countyFIPS", "wave_number"],
        "step2_monthly_median_reff.csv": ["onset_month"], "step3_analysis_table.csv": ["countyFIPS"],
        "step3_main_results.csv": ["factor"], "step3_robustness.csv": ["variant", "factor"],
        "step3_weight_robustness.csv": ["variant", "factor"], "step3_variance_decomposition.csv": ["model"],
        "step4_analysis_table.csv": ["countyFIPS"], "step4_main_results.csv": ["factor"],
        "step4_secondary_factors.csv": ["factor"], "step4_deaths_per100k.csv": ["factor"],
        "step4_decomposition.csv": ["model"], "step4_robustness.csv": ["variant", "factor"],
        "step4_pooled_by_month.csv": ["onset_month"], "step4_mutually_adjusted.csv": ["outcome", "factor"]}


def isnum(x):
    return pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)


def sig2(v):
    return float(f"{v:.2g}")


def diff(old, new):
    rows = []
    for f, key in KEYS.items():
        a = pd.read_csv(os.path.join(old, f), dtype={"countyFIPS": str})
        b = pd.read_csv(os.path.join(new, f), dtype={"countyFIPS": str})
        m = a.merge(b, on=key, how="outer", suffixes=("_old", "_new"), indicator="side")
        for _, r in m[m.side != "both"].iterrows():
            rows.append({"file": f, "key": " | ".join(str(r[k]) for k in key), "column": "<row>",
                         "old": "present" if r.side == "left_only" else "absent",
                         "new": "present" if r.side == "right_only" else "absent"})
        both = m[m.side == "both"].reset_index(drop=True)
        for c in [c for c in a.columns if c not in key and c in b.columns]:
            o, n = both[c + "_old"], both[c + "_new"]
            numeric = isnum(o) and isnum(n)
            if numeric:
                on, nn = o.astype(float), n.astype(float)
                diff = ~np.isclose(on, nn, rtol=1e-9, atol=0, equal_nan=True)
            else:
                diff = (~((o.isna() & n.isna()) | (o.astype(str) == n.astype(str)))).to_numpy()
            for i in np.flatnonzero(diff):
                r = both.iloc[i]
                rec = {"file": f, "key": " | ".join(str(r[k]) for k in key), "column": c,
                       "old": o.iloc[i], "new": n.iloc[i]}
                if numeric:
                    ov, nv = on.iloc[i], nn.iloc[i]
                    fin = np.isfinite(ov) and np.isfinite(nv)
                    rec["rel_change"] = (nv - ov) / abs(ov) if fin and ov != 0 else np.nan
                    rec["second_sig_fig_changed"] = (sig2(ov) != sig2(nv)) if fin else True
                rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    old, new, out = sys.argv[1:4]
    d = diff(old, new)
    d.to_csv(out, index=False)
    print(d.groupby("file").size().to_string())
