import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# dictionary of data to hold sample complexities

data = {'Network': ["MLP", "MLP", "RNN", "RNN", "MMC", "MMC"],
        'Variant': ['I', 'II', 'I', 'II', 'I', 'II'],
        'Interactions': [2000, 3000, 2000, 3000, 125, 250]}

df = pd.DataFrame.from_dict(data)
flatui = ["#AFDDE8", "#E8C6AF", "#FFEDAA"]
sns.set_palette(flatui)
fig = plt.figure()
sns.set(style="dark")
sns.set_context("paper")
g = sns.factorplot(x='Network', y='Interactions', hue='Variant', data=df, kind='bar')
plt.ylabel(r'Interactions ($e^{+02}$)', fontsize=12)
plt.xlabel("Network", fontsize=12)
g.despine(left=True)
plt.setp(g._legend.get_title(), fontsize=11)
plt.savefig("interactions.pdf", bbox_inches="tight")