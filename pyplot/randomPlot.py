import matplotlib.pyplot as plt
import numpy as np

dotx = np.random.rand(100) * 100
doty = np.random.rand(100) * 10

fig, ax = plt.subplots()

# ax.plot(dotx, doty)
# ax.plot(dotx, '--')
# ax.plot(dotx, '.')
# ax.plot(dotx, '.', color='#238374', label='20')

cValue = ['r','y','g','b','r','y','g','b','r'] 
ax.scatter(dotx, doty, c=cValue, marker='x')

ax.grid()

plt.title('ahahah')
plt.show()
