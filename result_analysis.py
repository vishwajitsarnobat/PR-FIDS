import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_comparison(results_file="simulation_results.csv"):
    """
    Loads simulation results and generates a formal plot comparing the accuracy
    of different scenarios over federated learning rounds.
    """
    try:
        df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"Error: The results file '{results_file}' was not found.")
        print("Please run main.py to generate the results first.")
        return

    # --- Plotting Configuration ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(12, 8))

    # Create the plot
    ax = sns.lineplot(
        data=df,
        x="round",
        y="accuracy",
        hue="scenario",
        style="scenario",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8
    )

    # --- Formatting for a Formal, Research-Grade Appearance ---
    ax.set_title("Federated IDS Performance Under Different Scenarios", fontsize=20, weight='bold')
    ax.set_xlabel("Federated Learning Round", fontsize=16)
    ax.set_ylabel("Global Model Accuracy (%)", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[label.replace('_', ' ').title() for label in labels], title="Scenario", fontsize=12, title_fontsize=14)

    plt.tight_layout()
    
    # Save the plot
    output_filename = "accuracy_comparison_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved as {output_filename}")
    plt.show()

if __name__ == '__main__':
    # This allows you to generate plots by simply running this script,
    # assuming simulation_results.csv already exists.
    plot_accuracy_comparison()