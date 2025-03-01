# Part a:
# dx/dt = σ(y - x) , dy/dt = x(ρ - z) - y , dz/dt = xy - βz
"""
This can represent atmospheric convection. x and y represent the horizontal and vertical velocities of the fluid, while z represents the temperature 
difference between the rising and sinking air. The parameters σ, ρ, and β represent the Prandtl number, Rayleigh number, and the geoetric factor of the system.
"""

# Part b:

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio
import os

# Define the Lorenz system
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters for Lorenz system
sigma = 10
rho = 48
beta = 3

# Initial conditions
x0, y0, z0 = 1.0, 1.0, 1.0  # Starting point for the system

# Time range for integration (extended to 20)
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Reduced number of steps for faster video

# Solve the Lorenz system
solution = solve_ivp(lorenz, t_span, [x0, y0, z0], args=(sigma, rho, beta), t_eval=t_eval, rtol=1e-8)

# Extract the solution
xa, ya, za = solution.y

# Create a video writer with imageio
video_writer = imageio.get_writer('lorenz_video.mp4', fps=30)

# Plot the Lorenz attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("Lorenz Attractor")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Create the frames and write them to the video
for i in range(0, len(t_eval), 100):  # Create a frame every 100 time steps (adjusted)
    ax.cla()  # Clear the previous plot
    ax.plot(xa[:i], ya[:i], za[:i], lw=0.8, color='b')  # Plot the evolution up to time i
    
    # Capture the current frame and convert to image
    plt.draw()
    plt.pause(0.01)
    
    # Save the frame to a temporary image
    plt.savefig('temp_frame.png', dpi=100)
    
    # Read the image and append it to the video
    frame = imageio.imread('temp_frame.png')
    video_writer.append_data(frame)

# Close the video writer and clean up
video_writer.close()
plt.close(fig)

# Remove temporary frame image
os.remove('temp_frame.png')

print("Video has been saved as 'lorenz_video.mp4'.")
