# scripts/visualize_results.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_correlations():
    df = pd.read_csv("data/cleaned_house_data.csv")

    # Compute correlations
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("results/correlations_heatmap.png")
    plt.close()
    print("📊 Heatmap saved to results/correlations_heatmap.png")

if __name__ == "__main__":
    visualize_correlations()
