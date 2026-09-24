"""Phase 1: load, validate, and split the data; save a summary table and EDA figures."""

from __future__ import annotations

from oncolens import config
from oncolens.data import load_raw_dataframe, load_splits, summary_table, validate_dataframe
from oncolens.io_utils import write_json
from oncolens.plots_eda import (
    plot_class_balance,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_pca_scatter,
)
from oncolens.style import apply_style, save_figure


def main() -> None:
    """Run the data phase end to end."""
    config.ensure_dirs()
    apply_style()

    df = load_raw_dataframe()
    problems = validate_dataframe(df)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))

    train, test = load_splits()
    overview = {
        "n_samples": len(df),
        "n_features": df.shape[1] - 1,
        "n_malignant": int(df[config.TARGET_COL].sum()),
        "n_benign": int((df[config.TARGET_COL] == 0).sum()),
        "malignant_share": round(float(df[config.TARGET_COL].mean()), 4),
        "n_train": len(train),
        "n_test": len(test),
        "train_malignant_share": round(float(train[config.TARGET_COL].mean()), 4),
        "test_malignant_share": round(float(test[config.TARGET_COL].mean()), 4),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "validation_problems": problems,
    }
    write_json(overview, config.RESULTS_DIR / "data_overview.json")
    # Descriptive statistics use the training split only, so nothing about the test set leaks
    # into any decision made while exploring.
    summary_table(train).to_csv(config.RESULTS_DIR / "data_summary_train.csv")

    save_figure(plot_class_balance(df), config.FIGURES_DIR / "01_class_balance.png")
    save_figure(plot_correlation_heatmap(train), config.FIGURES_DIR / "02_correlation_heatmap.png")
    save_figure(plot_pca_scatter(train), config.FIGURES_DIR / "03_pca_scatter.png")
    save_figure(
        plot_feature_distributions(train, ["worst concave points", "worst perimeter", "mean texture"]),
        config.FIGURES_DIR / "04_feature_distributions.png",
    )
    print(f"Data OK: {overview}")


if __name__ == "__main__":
    main()
