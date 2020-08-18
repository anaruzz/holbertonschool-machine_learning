#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

ind = np.arange(3)
f = list(fruit)

p1 = plt.bar(ind, f[0], 0.5, color='red')
p2 = plt.bar(ind, f[1], 0.5, color='yellow',bottom=f[0])
p3 = plt.bar(ind, f[2], 0.5, color='#ff8000', bottom=f[0]+f[1])
p4 = plt.bar(ind, f[3], 0.5, color='#ffe5b4', bottom=f[0] + f[1] + f[2])

plt.legend((p1[0], p2[0], p3[0], p4[0]),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.title('Number of Fruit per Person')
plt.ylim(0, 80)
plt.ylabel('Quantity of Fruit')
plt.xticks(ind, ('Farrah', 'Fred', 'Felicia'))

plt.show()
