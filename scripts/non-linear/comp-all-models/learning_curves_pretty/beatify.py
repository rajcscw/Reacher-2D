import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dfs_to_read = ["selected_df/diff_false_learning_curve_df", "selected_df/diff_true_learning_curve_df"]

flatui = ["#AFDDE8", "#E8C6AF", "#FFEDAA"]
sns.set_palette(flatui)

for df_ in dfs_to_read:
    df = pd.read_pickle(df_)
    fig = plt.figure()
    sns.set(style="dark")
    sns.set_context("paper")
    #sns.set_palette("Set1")
    sns.lineplot(data=df, x="Iteration", hue="strategy", y="Episodic Total Reward", ci="sd", err_style="band")
    plt.ylabel("Episodic Total Reward", fontsize=12)
    plt.xlabel("Iteration", fontsize=12)
    plt.legend(loc=4, fontsize=12)
    plt.savefig(df_+".pdf")