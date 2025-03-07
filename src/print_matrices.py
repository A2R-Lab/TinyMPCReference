import numpy as np
from quadrotor import QuadrotorDynamics
from tinympc import TinyMPC

# Create the quadrotor dynamics object
quad = QuadrotorDynamics()

# Define a reference hover state (13-state with quaternion)
x_ref = np.zeros(13)
x_ref[3] = 1.0  # Quaternion w component = 1 (identity orientation)

# Define reference hover input
u_ref = quad.hover_thrust

# Get linearized dynamics matrices
A_full, B_full = quad.get_linearized_dynamics(x_ref, u_ref)

# Since your implementation uses a 12-state model, we need to convert to 12 states
# (the linearized matrices should already be in the right format with 12 states)
print("A matrix (12x12):")
print(A_full)
print("\nB matrix (12x4):")
print(B_full)

# Create cost matrices
Q = np.diag([10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1])  # Higher weights for position/attitude
R = np.diag([0.1, 0.1, 0.1, 0.1])  # Control penalty

# Create TinyMPC object with these matrices
# This will compute Kinf, Pinf, and other cache terms
tinympc = TinyMPC(A_full, B_full, Q, R, Nsteps=10, rho=85.0)

# Print the computed matrices
print("\nKinf matrix (4x12):")
print(tinympc.cache['Kinf'])
print("\nPinf matrix (12x12):")
print(tinympc.cache['Pinf'])
print("\nC1 matrix (4x4):")
print(tinympc.cache['C1'])
print("\nC2 matrix (12x12):")
print(tinympc.cache['C2'])