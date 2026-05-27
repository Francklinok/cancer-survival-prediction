import matplotlib.pyplot as plt

def flag_plot(flag, n_was_missing, n_total, cross):
    """
    Visualizes the impact of an imputation flag on the target variable.
    
    - Pie chart: Shows the proportion of imputed vs. original values.
    - Bar plot: Shows the distribution of the target (e.g., TreatmentResponse) 
      segmented by the presence/absence of the missing value flag.
    """
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
    pivot = cross.pivot(index="TreatmentResponse", columns=flag, values="pct")
    pivot.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"], edgecolor="white")
    
    axes[1].set_title(f"Response by {flag} status")
    axes[1].set_ylabel("% patients")
    axes[1].set_xlabel("Treatment Response")
    axes[1].legend(title="Is Missing?", labels=["No", "Yes"], loc="upper right")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()