import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Part a:

# Julia set function
def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter=256):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    img = np.zeros(Z.shape, dtype=int)
    points = []
    
    for i in range(max_iter):
        mask = np.abs(Z) < 2
        img[mask] = i
        Z[mask] = Z[mask] ** 2 + c
    
    for i in range(width):
        for j in range(height):
            if img[j, i] == max_iter - 1:
                points.append([x[i], y[j]])
    
    return img, np.array(points)

xmin, xmax, ymin, ymax = -1.5, 1.5, -1, 1
width, height = 800, 800
c = complex(-0.7, 0.356)

julia, points = julia_set(xmin, xmax, ymin, ymax, width, height, c)


# Plotting the Julia set
plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap="inferno")
plt.colorbar(label="Iterations")
plt.title("Julia Set for c = -0.7 + 0.356i")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.savefig("julia_set.png")

# Part b:

# Convex hull area function
def convex_hull_area(points):
    if len(points) < 3:
        return 0  # A hull cannot exist with fewer than 3 points
    hull = ConvexHull(points)
    return hull.volume  # ConvexHull.volume gives the area in 2D

area = convex_hull_area(points)

# Plotting the Julia set with convex hull overlaid
plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap="inferno")
plt.colorbar(label="Iterations")
plt.title(f"Julia Set for c = -0.7 + 0.356i\nConvex Hull Area = {area:.4f}")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")

# Plot convex hull
hull = ConvexHull(points)
hull_points = points[hull.vertices]

# Flip the y-coordinates to match the image coordinates and shift by 1 unit down
hull_points[:, 1] = ymax - hull_points[:, 1] - 1  # Shifting by 1 unit down

plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', lw=2)  # Red line for convex hull

plt.savefig("julia_set_area.png")


# Part c: Using matplotlib to find contours

def contour_area(img):
    contours = plt.contour(img, levels=[0.5], colors='red')
    max_area = 0
    for collection in contours.collections:
        for path in collection.get_paths():
            # Calculate the area of each closed contour using the shoelace formula
            vertices = path.vertices
            x = vertices[:, 0]
            y = vertices[:, 1]
            
            # Shoelace formula for polygon area
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            max_area = max(max_area, area)
    return max_area


# Compute contour area
area = contour_area(julia)

plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap="inferno")
plt.colorbar(label="Iterations")
plt.title(f"Julia Set for c = -0.7 + 0.356i\nContour Area = {area:.4f}")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.savefig("julia_set_contour_area.png")


# Part d: Box-counting method to estimate the fractional dimension

def box_counting(img, min_size=2, max_size=100):
    sizes = []
    counts = []
    
    # Loop over different box sizes (ϵ)
    for size in range(min_size, max_size + 1):
        count = 0
        for i in range(0, img.shape[0], size):
            for j in range(0, img.shape[1], size):
                # Check if there is any non-zero point in the box
                if np.any(img[i:i+size, j:j+size] > 0):
                    count += 1
        sizes.append(size)
        counts.append(count)
    
    # Compute log-log scale for box-counting
    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts))
    
    # Fit a line to log-log data to get the fractal dimension (slope)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = coeffs[0]
    
    return fractal_dimension

# Estimate the fractal dimension using the box-counting method
fractal_dim = box_counting(julia)

plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap="inferno")
plt.colorbar(label="Iterations")
plt.title(f"Julia Set for c = -0.7 + 0.356i\nFractal Dimension = {fractal_dim:.4f}")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.savefig("julia_set_fractal_dimension.png")

print(f"Estimated fractal dimension: {fractal_dim:.4f}")

# Estimated fractal dimension: 1.9462