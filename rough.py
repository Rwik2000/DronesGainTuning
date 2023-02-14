import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

np.random.seed(0)
a = np.random.rand(4,4)
print(a)
# i, j = np.ogrid[:4, :5]
# x = 10*i + j

v = sliding_window_view(a,(2,4))

print(v)

# print(x)