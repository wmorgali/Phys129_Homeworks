import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Part a:

# Takes a value of r and returns the fixed points of the logistic map
def logistic_map_fixed_points(r):
    x = sp.Symbol('x')
    eq = sp.Eq(r * x * (1 - x), x)
    fixed_points = sp.solve(eq, x)
    return fixed_points

# Takes a value of r and the corresponding fixed points and returns the stability of each fixed point
def stability_analysis(r, fixed_points):
    x = sp.Symbol('x')
    f_prime = sp.diff(r * x * (1 - x), x)
    stability = {}
    
    for fp in fixed_points:
        derivative_at_fp = f_prime.subs(x, fp)
        stability[fp] = derivative_at_fp
        
        if abs(derivative_at_fp) < 1:
            stability[fp] = "Stable"
        else:
            stability[fp] = "Unstable"
    
    return stability

# Test for r = 1, 2, 3, 4
r_values = [1, 2, 3, 4]
for r in r_values:
    print(f"For r = {r}:")
    fixed_points = logistic_map_fixed_points(r)
    stability = stability_analysis(r, fixed_points)
    for fp, stab in stability.items():
        print(f"  Fixed Point: {fp}, Stability: {stab}")
    print()


# Part b:

# Function that iterates the log map. Stops after 1000 iterations or when the difference between consecutive values is less than 1e-6
def iterate_logistic_map(r, x0, threshold=1e-6, max_iter=1000):
    prev_x = x0
    trajectory = [prev_x]
    for _ in range(max_iter):
        next_x = r * prev_x * (1 - prev_x)
        trajectory.append(next_x)
        if abs(next_x - prev_x) < threshold:
            break
        prev_x = next_x
    return trajectory

# New r values
r_valuesb = [1, 2, 3, 3.5, 3.8, 4.0]

# Prints last 10 values of the iterated logistic map for each r value
for r in r_valuesb:
    trajectory = iterate_logistic_map(r, 0.2)
    print(f"For r = {r}:")
    print(f"  Iterated Values: {trajectory[-10:]}")
    print()


# Part c:

# Initial x0 values
initial_conditions = [0.1, 0.2, 0.3, 0.5]

# Plot the time series for each r value
for r in r_values:
    plt.figure(figsize=(8, 5))
    for x0 in initial_conditions:
        trajectory = iterate_logistic_map(r, x0)
        plt.plot(trajectory, label=f"x0={x0}")
    plt.xlabel("Iteration")
    plt.ylabel("x_n")
    plt.title(f"Logistic Map Time Series for r = {r}")
    plt.legend()
    plt.savefig(f"logistic_map_{r}.png")

# The time series look very different for each r value. Some values of r have stable fixed points, while others have chaotic behavior.

# Part d:

# New iterate logistic map function that returns the trajectory after discarding the first 500 iterations
def new_iterate_logistic_map(r, x0, num_iter=1000, discard=500):
    x = x0
    trajectory = []
    for _ in range(num_iter):
        x = r * x * (1 - x)
        if _ >= discard:
            trajectory.append(x)
    return trajectory

# New r values
r_valuesd = np.linspace(0, 4, 1000)
x0 = 0.2
bifurcation_data = []

for r in r_valuesd:
    trajectory = iterate_logistic_map(r, x0)
    bifurcation_data.extend([(r, x) for x in trajectory])

# Plot the bifurcation diagram
r_vals, x_vals = zip(*bifurcation_data)
plt.figure(figsize=(10, 6))
plt.scatter(r_vals, x_vals, s=0.1, color='black')
plt.xlabel("r (Control Parameter)")
plt.ylabel("xn (Population)")
plt.title("Bifurcation Diagram of the Logistic Map with Fixed Points")
plt.savefig("bifurcation_diagram.png")

for r in [1, 2, 3, 3.5, 3.8, 4.0]:
    print(f"For r = {r}:")
    fixed_points = logistic_map_fixed_points(r)
    stability = stability_analysis(r, fixed_points)
    for fp, stab in stability.items():
        print(f"  Fixed Point: {fp}, Stability: {stab}")
    print()


# Part e:

# Modified logistic map function with an additional parameter gamma
def modified_logistic_map(r, gamma, x0, num_iter=1000, discard=500):
    x = x0
    trajectory = []
    for _ in range(num_iter):
        x = r * x * (1 - x**gamma)
        if _ >= discard:
            trajectory.append(x)
    return trajectory


def find_first_bifurcation(r_valuesd, gamma_values, x0=0.2):
    bifurcation_points = []
    for gamma in gamma_values:
        for r in r_valuesd:
            trajectory = modified_logistic_map(r, gamma, x0)
            unique_points = set(np.round(trajectory[-100:], 6))
            if len(unique_points) > 1:
                bifurcation_points.append((gamma, r))
                break
    return bifurcation_points

gamma_values = np.linspace(0.5, 1.5, 100)
bifurcation_points = find_first_bifurcation(r_valuesd, gamma_values)

gamma_vals, r_vals = zip(*bifurcation_points)
plt.figure(figsize=(8, 5))
plt.plot(gamma_vals, r_vals, marker='o', linestyle='-', color='blue')
plt.xlabel("Gamma (γ)")
plt.ylabel("First Bifurcation Point (r)")
plt.title("First Bifurcation Point as a Function of γ")
plt.savefig("bifurcation_gamma.png")