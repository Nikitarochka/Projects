import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp
sample1 = np.random.normal(loc=10, scale=11, size=300)
sample2 = np.random.normal(loc=10, scale=10, size=300)
sample = np.hstack((sample1, sample2)) # Создаем объединенную выборку из №1 и №2
sns.histplot(sample1,bins = 10,color='red')
sns.histplot(sample2,bins = 10,color = 'green')
sns.histplot(sample,bins = 10)
plt.show()
print(sp.ks_2samp(sample1, sample2))