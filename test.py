import torch
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

def himmelblau(x):
    return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 -7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6 , 0.1)
print('x, y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X, Y range:', x.shape, y.shape)
Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = fig.add_subplot(111, projection='3d')  # Corrected this line
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# Initialize the tensor with gradients
x = torch.tensor([4.0, 0.0], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)

# Optimization loop
for step in range(20000):
    pred = himmelblau(x)  # Calculate the function value
    optimizer.zero_grad()  # Zero the gradients
    pred.backward()  # Backpropagate
    optimizer.step()  # Update the parameters

    # Print progress
    if step % 2000 == 0:  # Adjust this to print more frequently
        print('step {}: x={}, f(x)={}'.format(step, x.tolist(), pred.item()))

# Final output
print('Final result: x={}, f(x)={}'.format(x.tolist(), himmelblau(x).item()))
