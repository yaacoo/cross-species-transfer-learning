import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
sns.set_palette("Set2")

# Load and concatentate the results tables from the results directory:
path="../results/"
files = os.listdir(path)
files = [f for f in files if f.endswith(".csv")]
results = pd.concat([pd.read_csv(path+f) for f in files])
# Change the first column to "Identity":
results = results.rename(columns={results.columns[0]: "Identity"})
# Change the results to a tidy long format:
results = results.melt(id_vars=["Identity", "Tissue", "Algorithm", "Type"], var_name="Metric", value_name="Value")

# Capitalize the first letter of the Identity column (for the plots):
results['Identity'] = results['Identity'].str.capitalize()
# Remove the "weighted avg" rows because they are influenced by the majority class, which is easier to predict:
results = results[results['Identity'] != 'Weighted avg']
# Keep the macro average because it is more stringent when there is class imbalance:
results = results.replace({'Identity': {'Macro avg': 'Average (macro)'}})

# In the column Type, replace "Source on target" with "Without transfer learning":
results['Type'] = results['Type'].replace("Source on target", "Without transfer learning")
# sort the results table by the Type column:
results = results.sort_values(by=['Type', 'Identity'], ascending=False)

# Plot the results for each tissue:
for Algotithm_to_plot in results.Algorithm.unique():
    print(Algotithm_to_plot)
    for tissue_to_plot in results.Tissue.unique():
        metric_to_plot="f1-score"
        # Create a bar plot of metric_to_plot grouped by the Type:
        sns.barplot(x="Value", y="Identity", hue="Type", 
            data=results[(results.Tissue==tissue_to_plot) & (results.Metric==metric_to_plot) & (results.Algorithm==Algotithm_to_plot) & (results.Identity != "Accuracy")])
        # Edit the tissue name to add spaces before the capital letters:
        tissue_to_plot_spaces = re.sub(r"([A-Z])", r" \1", tissue_to_plot).strip()
        # Set the title of the plot
        plt.title(f"{tissue_to_plot_spaces}: {Algotithm_to_plot}")
        # Set the letend outside the plot:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # Set the x-axis label:
        plt.xlabel(metric_to_plot)
        # Set the y-axis label:
        plt.ylabel("Cell Type")
        # Save the plot as PDF:
        plt.savefig(f"../figures/{metric_to_plot}_{tissue_to_plot}_{Algotithm_to_plot}.pdf", bbox_inches='tight')
        plt.close()

# Change all the rissue names to add spaces before the capital letters:
results['Tissue'] = results['Tissue'].str.replace(r"([A-Z])", r" \1").str.strip()

# Create an f1 summary table that conatins only the f1 average (macro), for plotting:
results_f1score = results[(results.Metric=="f1-score") & (results.Identity=="Average (macro)")].copy()

# plot the f1-score summary for each algorithm, by tissue:
for Algotithm_to_plot in results_f1score.Algorithm.unique():
    sns.barplot(x="Value", y="Tissue", hue="Type", data=results_f1score[results_f1score.Algorithm==Algotithm_to_plot])
    # Set the title of the plot
    plt.title(Algotithm_to_plot)
    # Set the legend outside the plot:
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Set the x-axis label:
    plt.xlabel("F1-score Average (macro)")
    # Set the y-axis label:
    plt.ylabel("")
    # Save the plot as PDF:
    plt.savefig(f"../figures/summary_f1score_{Algotithm_to_plot}.pdf", bbox_inches='tight')
    plt.close()


# Summarize the change in f1-score:
change_f1score = results[(results.Metric=="f1-score") & (results.Type!="Target on target") & (results.Identity!="Accuracy") & (results.Identity!="Average (macro)")].copy()
# Change to a wide format:
change_f1score = change_f1score.pivot_table(index=["Identity", "Tissue", "Algorithm"], columns="Type", values="Value")
change_f1score = change_f1score.reset_index()
# Acc a culumn to the change_f1score table that contains the change in f1-score:
change_f1score["Change in f1-score"] = change_f1score["With transfer learning"] - change_f1score["Without transfer learning"]
# Sort the change_f1score table by the "Change in f1-score" column:
change_f1score = change_f1score.sort_values(by="Change in f1-score", ascending=False)

# Create a bar of "total combined" for the plot:
change_f1score_total = change_f1score.copy()
# Change all of the Tissue values to "Total Combined":
change_f1score_total["Tissue"] = "Total Combined"
# Concatenate the change_f1score table with the change_f1score_total table:
change_f1score = pd.concat([change_f1score, change_f1score_total])

# Create a bar plot of change in f1-score grouped by tissue:
sns.catplot(x="Change in f1-score", y="Tissue", hue="Algorithm", col="Algorithm", data=change_f1score, kind="bar", height=7, aspect=.8)
# Set the x-axis label:
plt.xlabel("Change in f1-score")
# Set the y-axis label:
plt.ylabel("")
plt.savefig(f"../figures/summary_change_f1score.pdf", bbox_inches='tight')
plt.close()