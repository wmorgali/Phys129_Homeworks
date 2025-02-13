import os
import numpy as np
import matplotlib.pyplot as plt

# Define directories
input_dir = "Local_density_of_states_near_band_edge"
output_dir = os.path.join(input_dir, "local_density_of_states_heatmap")
heightmap_output_dir = os.path.join(input_dir, "local_density_of_states_height")

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(heightmap_output_dir, exist_ok=True)

# Define sub-region (example: center 10x10 region) for part c.
subregion_start_x, subregion_end_x = 20, 30
subregion_start_y, subregion_end_y = 20, 30

average_ldos_values = []
indices = []

# Process each LDOS file
for filename in os.listdir(input_dir):
    if filename.startswith("local_density_of_states_for_level_") and filename.endswith(".txt"):
        filepath = os.path.join(input_dir, filename)
        
        # Load LDOS data
        data = np.loadtxt(filepath, delimiter=',')


        # Extract sub-region and compute average LDOS
        subregion = data[subregion_start_y:subregion_end_y, subregion_start_x:subregion_end_x]
        average_ldos = np.mean(subregion)
        
        # Extract index from filename
        file_index = int(filename.split("_")[-1].split(".")[0])
        average_ldos_values.append(average_ldos)
        indices.append(file_index)
        
        # Generate heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(data, cmap='inferno', aspect='auto', origin='lower')
        plt.colorbar(label='LDOS Intensity')
        
        
        plt.title(f"LDOS Heatmap (Level {file_index})")
        
        # Save heatmap
        output_filepath = os.path.join(output_dir, f"ldos_heatmap_{file_index}.png")
        plt.savefig(output_filepath, dpi=300)
        plt.close()
        
        # Generate 2D surface plot
        plt.figure(figsize=(6, 5))
        plt.contourf(data, cmap='viridis', levels=50)
        plt.colorbar(label='LDOS Intensity')
        plt.title(f"LDOS Surface Plot (Level {file_index})")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        
        # Save surface plot
        heightmap_filepath = os.path.join(heightmap_output_dir, f"ldos_surface_{file_index}.png")
        plt.savefig(heightmap_filepath, dpi=300)
        plt.close()


# Plot average LDOS changes across indices and plot
plt.figure(figsize=(8, 6))
plt.plot(sorted(indices), [x for _, x in sorted(zip(indices, average_ldos_values))], marker='o', linestyle='-')
plt.xlabel("Energy Level Index")
plt.ylabel("Average LDOS in Subregion")
plt.title("Average LDOS Changes Across Energy Levels")
plt.grid()
plt.savefig(os.path.join(input_dir, "average_ldos_trend.png"), dpi=300)
plt.show()
print("Heatmaps and surface plots generated and saved successfully!")
