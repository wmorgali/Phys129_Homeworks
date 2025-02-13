"""
import numpy as np
from scipy.integrate import nquad
from scipy.linalg import det, inv

# Integral and closed-form expression code
def gaussian_integral_verification(A, w):
    N = len(w)  # Dimension

    # Ensure A is positive definite
    try:
        np.linalg.cholesky(A)  # Check if A is positive definite
    except np.linalg.LinAlgError:
        print("Matrix A is not positive definite. Adjusting...")
        A += N * np.eye(N)  # Make A strictly positive definite
    
    A_inv = inv(A)  # Inverse of matrix A
    
    # Define the integrand
    def integrand(*v):
        v = np.array(v)
        exponent = -0.5 * v @ A @ v + v @ w
        return np.exp(exponent)
    
    # Compute numerical integration using nquad
    integral_numerical, error = nquad(integrand, [(-np.inf, np.inf)] * N)

    # Compute the closed-form expression
    integral_analytical = np.sqrt((2 * np.pi) ** N * det(A_inv)) * np.exp(0.5 * w @ A_inv @ w)
    
    return integral_numerical, integral_analytical

# Part b, test matrices.
# Example usage
if __name__ == "__main__":
    N = 3  # Dimension
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])  # Example matrix
    w = np.array([1, 2, 3])  # Example vector

    '''
    num_result, analytical_result = gaussian_integral_verification(A, w)
    print(f"Numerical integral: {num_result}")
    print(f"Analytical result: {analytical_result}")
    
    A2 = np.array([[4, 2, 1], [2, 1, 3], [1, 3, 6]])  # Example matrix
    num_result2, analytical_result2 = gaussian_integral_verification(A2, w)
    print(f"Numerical integral: {num_result2}")
    print(f"Analytical result: {analytical_result2}")
    '''
# For A:
# Numerical integral: 4.275823659021463
# Analytical result: 4.275823659011516

# For A', the determinant of A' inverse is negative which means you cant take the square root.
# This makes the closed form expression invalid.
"""

# Part c
import numpy as np

def analytic_moment(A, w, powers):
    """Compute moments analytically using mean and covariance."""
    S = np.linalg.inv(A)  # Covariance matrix
    mu = S @ w            # Mean vector
    indices = np.array([i for i, p in enumerate(powers) for _ in range(p)])
    n = len(indices)
    
    if n % 2 != 0:
        return 0.0  # Odd moments vanish for zero-mean Gaussians
    
    if n == 1:
        return mu[indices[0]]
    elif n == 2:
        i, j = indices
        return mu[i] * mu[j] + S[i, j]
    else:
        i = indices[0]
        total = 0.0
        for j in range(1, len(indices)):
            pair_cov = S[i, indices[j]]
            remaining = np.concatenate([indices[1:j], indices[j+1:]])
            total += pair_cov * analytic_moment(A, w, remaining)
        return total

def verify_moment(A, w, powers, label=""):
    """Compare analytic moments and display results."""
    analytic_val = analytic_moment(A, w, powers)
    print(f"{label}: {analytic_val} (Closed-form: {format_closed_form(A, w, powers)})")

def format_closed_form(A, w, powers):
    """Generate a closed-form expression in terms of A and w."""
    S = np.linalg.inv(A)
    mu = S @ w
    indices = [i for i, p in enumerate(powers) for _ in range(p)]
    n = len(indices)
    
    if n == 1:
        return f"sum_j (A^(-1))_{{{indices[0]}j}} w_j"
    elif n == 2:
        i, j = indices
        return f"(A^(-1))_{{{i}{j}}} + (sum_k (A^(-1))_{{{i}k}} w_k) (sum_m (A^(-1))_{{{j}m}} w_m)"
    else:
        return "Higher-order expression (follows Wick's theorem)"

# Given matrix A and vector w
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])
w = np.array([1, 2, 3])

print("\nComputed Moments with Closed-Form Expressions:")
verify_moment(A, w, [1, 0, 0], "<v1>")
verify_moment(A, w, [0, 1, 0], "<v2>")
verify_moment(A, w, [0, 0, 1], "<v3>")
verify_moment(A, w, [1, 1, 0], "<v1 v2>")
verify_moment(A, w, [0, 1, 1], "<v2 v3>")
verify_moment(A, w, [1, 0, 1], "<v1 v3>")
verify_moment(A, w, [2, 1, 0], "<v1^2 v2>")
verify_moment(A, w, [0, 2, 1], "<v2^2 v3>")
verify_moment(A, w, [2, 2, 0], "<v1^2 v2^2>")
verify_moment(A, w, [0, 2, 2], "<v2^2 v3^2>")
