import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import vectormath as vmath

# Load input image
original_img = np.asarray(mpimg.imread('grid.png'))

# Define warp function
def warp_point(point):
    yn = point[0]
    xn = point[1]
    xo = math.sin(yn*math.pi) * 0.1 # 10% horizontal warp
    yo = 0# math.sin(xn*math.pi) * 0.1 # 10% horizontal warp
    return (yn + yo, xn + xo)

# Create warped image
shape = original_img.shape
warped_img = np.zeros(shape)
for y in range(shape[0]):
    yn = y / (shape[0] - 1)
    for x in range(shape[1]):
        xn = x / (shape[1] - 1)
        nx = int(-warp_point((yn, -xn))[1] * shape[1]) # To shift right we need to sample to the left
        ny = int(-warp_point((-yn, xn))[0] * shape[0]) # To shift right we need to sample to the left
        for c in range(shape[2]):
            sample = original_img[ny, nx, c] if nx >= 0 and ny >= 0 and nx < shape[1] and ny < shape[0] else 0
            warped_img[y, x, c] = sample

# Create warped point grid
grid_h = 5
grid_w = 5
grid_x = []
grid_y = []
grid = np.zeros((grid_w, grid_h, 2))
for yi in range(grid_h):
    for xi in range(grid_w):
        yn = yi / (grid_h - 1)
        xn = xi / (grid_w - 1)
        point = warp_point((yn, xn))
        grid_y.append((point[0] * shape[0]))
        grid_x.append((point[1] * shape[1]))
        grid[xi][yi] = (point[1] * shape[1], point[0] * shape[0])

# imgplot = plt.imshow(warped_img, zorder=1)
# plt.scatter(grid_x, grid_y, zorder=2)
# plt.show()
# quit()

# Unwarp the image
unwarped_img = np.zeros(shape)
for y in range(shape[0]):
    yn = y / shape[0]
    for x in range(shape[1]):
        xn = x / shape[1]
        yi = yn * (grid.shape[0] - 1)
        xi = xn * (grid.shape[1] - 1)
        yi0 = math.floor(yi)
        yi1 = yi0 + 1
        xi0 = math.floor(xi)
        xi1 = xi0 + 1
        ynn = yi - yi0
        xnn = xi - xi0
        p00 = vmath.Vector2(grid[xi0][yi0])
        p01 = vmath.Vector2(grid[xi0][yi1])
        p10 = vmath.Vector2(grid[xi1][yi0])
        p11 = vmath.Vector2(grid[xi1][yi1])
        t = (p10 - p00)
        b = (p11 - p01)
        pt = t.length * xnn * t.normalize() + p00
        pb = b.length * xnn * b.normalize() + p01
        ly = pb - pt
        p = ly.length * ynn * ly.normalize() + pt
        nx = int(p[0])
        ny = int(p[1])
        for c in range(shape[2]):
            sample = warped_img[ny, nx, c] if nx >= 0 and ny >= 0 and nx < shape[1] and ny < shape[0] else 0
            unwarped_img[y, x, c] = sample

# Show plots
imgplot = plt.imshow(unwarped_img, zorder=1)
plt.scatter(grid_x, grid_y, zorder=2)
plt.show()