import pandas as pd

temp = pd.read_csv('Data_1.csv')
print(temp)

print(temp['Month'])

temp['T'] = 0
print(temp)

import matplotlib.pyplot as plt

plt.plot(temp['A'], dashes=[6, 2])
plt.show()