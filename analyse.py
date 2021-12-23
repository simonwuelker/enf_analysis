# import pandas as pd
import matplotlib.pyplot as plt
# 
# df = pd.read_csv("data/fNew 2020 1.csv", index_col=0).head(100)
# 
# df.plot(kind = 'line')
# plt.show()
import numpy as np
from numpy.fft import fft

x = np.linspace(0, 100, 1000)
y = np.sin(x)
z = np.sin(2 * x)

# print(np.fft.fft(x + z))
plt.plot(np.abs(fft(x + z)))
plt.show()

