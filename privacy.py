from dp.dpsda.agm import get_epsilon, get_sigma
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

deltas = [1.e-3, 1.e-4, 1.e-5]
epsilons = [0.1, 1.0, 5.0, 10.0]
T = [1, 5, 10, 15, 20]
df = pd.DataFrame(index=pd.MultiIndex.from_product([deltas, epsilons, T], names=['delta', 'epsilon', 'T'])).reset_index()
sigmas = [0.] * df.shape[0]
total = [0.] * df.shape[0]
df['sigma'] = sigmas
df['total_eps'] = total
for delta in deltas:
    for eps in epsilons:
        for t in T:
            total_eps = get_epsilon(eps, t)
            print(f"epsilon: {eps}, T: {t}, total: {total_eps}")
            sigma = get_sigma(epsilon=eps, delta=delta, GS=1)
            condition = (df['epsilon'] == eps) & (df['delta'] == delta) & (df['T'] == t)
            df.loc[condition, 'total_eps'] = total_eps
            df.loc[condition, 'sigma'] = sigma



print(df)


# Pointplot 그리기
sns.set(style="whitegrid")
plt.subplots(figsize=(12, 8))
sns.pointplot(data=df, x='epsilon', y='total_eps', hue='T')
plt.title("Total epsilon")
plt.savefig("total.png")


sns.set(style="whitegrid")
plt.subplots(figsize=(12, 8))
sns.pointplot(data=df, x='epsilon', y='sigma', hue='delta')
plt.title("Sigma in one step")
plt.savefig("sigma.png")

df.to_excel(
    'data.xlsx',
    'data',
    float_format='%.8f',
    header=True,
    index=False,
    startcol=1
)