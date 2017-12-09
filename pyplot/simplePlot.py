import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

# ax.set(xlabel='time(s)', ylabel='valtage(mv)', title='about as simple')
ax.grid()

# fig.savefig('test.jpg')
plt.show()
