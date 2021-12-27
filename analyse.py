import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/fNew 2020 1.csv", index_col=0).head(100)

df.plot(kind = 'line')
print(df.max(axis=0))
plt.show()
