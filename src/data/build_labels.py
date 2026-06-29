"""Backward-compatible label-building module.

New code should import from `data.labels`.
"""

from __future__ import annotations

from data.labels import (
    FAUS,
    LABEL_MAPPING,
    RATERS,
    VALID_SCORES,
    LabelBuildSummary,
    build_labels,
    copy_impaired_images,
    fau_averages,
    parse_score,
    print_label_summary,
    round_half_up,
    welfare_label_from_mgs_row,
)


if __name__ == "__main__":
    from paths import IMAGES_MGS_DIR, MAIN_CSV, MGS_CSV

    df_mgs, summary = build_labels(
        MGS_CSV,
        MAIN_CSV,
        IMAGES_MGS_DIR,
        return_summary=True,
    )
    print_label_summary(df_mgs, summary)
