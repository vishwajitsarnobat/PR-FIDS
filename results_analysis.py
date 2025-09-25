import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_defense_effectiveness(results_file="simulation_results.csv"):
    """
    Loads all simulation results and generates the final "hero plot" demonstrating
    the effectiveness of the 3-layer defense system.
    """
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"Error: The results file '{results_file}' was not found.")
        print("Please run main.py with all scenarios to generate the results first.")
        return

    # --- Plotting Configuration ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(14, 9))

    # --- Strategic Color and Style Palette ---
    palette = {
        "baseline": "#003366",                  # Deep Blue
        "attack_label_flipping": "#FF8C00",      # Dark Orange
        "attack_backdoor": "#B22222",          # Firebrick Red
        "defended_label_flipping": "#228B22",   # Forest Green
        "defended_backdoor": "#4B0082"         # Indigo
    }
    
    dashes_map = {
        "baseline": (),
        "attack_label_flipping": (4, 2),
        "attack_backdoor": (4, 2),
        "defended_label_flipping": (),
        "defended_backdoor": ()
    }

    # Create the plot
    ax = sns.lineplot(
        data=df,
        x="round",
        y="accuracy",
        hue="scenario",
        style="scenario",
        dashes=dashes_map,
        palette=palette,
        markers=True,
        linewidth=2.5,
        markersize=9 # Increased size slightly for better visibility
    )

    # --- Formatting for a Formal, Research-Grade Appearance ---
    ax.set_title("Effectiveness of the Adaptive 3-Layer Defense System", fontsize=22, weight='bold', pad=20)
    ax.set_xlabel("Federated Learning Round", fontsize=16)
    ax.set_ylabel("Global Model Accuracy (%)", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.set_ylim(40, 95)
    ax.set_xticks(range(1, 11))

    # --- Customize Legend for Maximum Clarity ---
    handles, labels = ax.get_legend_handles_labels()
    
    legend_labels_map = {
        "baseline": "Baseline (No Attack)",
        "attack_label_flipping": "Undefended (Label Flipping)",
        "attack_backdoor": "Undefended (Backdoor)",
        "defended_label_flipping": "Defended (Label Flipping)",
        "defended_backdoor": "Defended (Backdoor)"
    }
    
    final_legend_labels = [legend_labels_map.get(lbl, lbl) for lbl in labels]

    ax.legend(handles=handles, labels=final_legend_labels, title="Scenario", fontsize=12, title_fontsize=14)

    plt.tight_layout()
    
    # Save the plot
    output_filename = "report_3_defense_effectiveness.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nReport 3 'Hero Plot' saved as {output_filename}")
    plt.show()

if __name__ == '__main__':
    plot_defense_effectiveness()