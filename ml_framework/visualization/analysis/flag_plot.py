import matplotlib.pyplot as plt

def flag_plot(flag, n_was_missing, n_total, cross):
    """
    Visualizes the impact of an imputation flag on the target variable.

    - Pie chart: Shows the proportion of imputed vs. original values.
    - Bar plot: Shows the distribution of the target variable, segmented
      by the presence/absence of the missing value flag.

    cross is expected to have exactly one non-flag, non-"pct" column —
    the target variable — as produced by missing_value.flag_analysis().
    """
    target_col = next(c for c in cross.columns if c not in (flag, "pct"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie chart — Proportion of the flag (missing vs. complete)
    axes[0].pie(
        [n_was_missing, n_total - n_was_missing],
        labels=["Was NaN\n(imputed)", "Complete"],
        autopct="%1.1f%%",
        colors=["#e74c3c", "#2ecc71"],
        startangle=90,
    )
    axes[0].set_title(f"Proportion of {flag}")

    # Bar plot — Target distribution by flag status
    pivot = cross.pivot(index=target_col, columns=flag, values="pct")
    
    color_map = {0: "#2ecc71", 1: "#e74c3c", False: "#2ecc71", True: "#e74c3c"}
    label_map = {0: "No", 1: "Yes", False: "No", True: "Yes"}
    bar_colors = [color_map.get(c, "#95a5a6") for c in pivot.columns]
    bar_labels = [label_map.get(c, str(c)) for c in pivot.columns]

    pivot.plot(kind="bar", ax=axes[1], color=bar_colors, edgecolor="white")

    axes[1].set_title(f"{target_col} by {flag} status")
    axes[1].set_ylabel("% patients")
    axes[1].set_xlabel(target_col)
    axes[1].legend(title="Is Missing?", labels=bar_labels, loc="upper right")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()