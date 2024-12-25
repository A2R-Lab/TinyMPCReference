import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from utils.visualization import visualize_trajectory, plot_iterations
from utils.traj_simulation import simulate_with_controller  # Changed import
from scipy.spatial.transform import Rotation as spRot
from utils.reference_trajectories import Figure8Reference
import matplotlib.pyplot as plt

def compute_tracking_error(x_all, trajectory, dt):
    """Compute L2 tracking error over the trajectory"""
    errors = []
    times = np.arange(len(x_all)) * dt
    
    for t, x in zip(times, x_all):
        # Get reference at this time
        x_ref = trajectory.generate_reference(t)
        
        # Compute position error (L2 norm of position difference)
        pos_error = np.linalg.norm(x[0:3] - x_ref[0:3])
        errors.append(pos_error)
    
    # Compute average and max error
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    return avg_error, max_error, errors

def main():
    # Create quadrotor instance
    quad = QuadrotorDynamics()

    # Initialize hover state (for linearization)
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    ug = quad.hover_thrust

    # Get linearized system
    A, B = quad.get_linearized_dynamics(xg, ug)

    # Initial state (start at origin)
    x0 = np.copy(xg)
    x0[0:3] = np.array([0.0, 0.0, 0.0])
    x0[3:7] = qg  # Start with hover attitude

    # Cost matrices (tuned for trajectory tracking)
    max_dev_x = np.array([
        0.01, 0.01, 0.01,    # position (tighter bounds)
        0.5, 0.5, 0.05,      # attitude
        0.5, 0.5, 0.5,       # velocity
        0.7, 0.7, 0.2        # angular velocity
    ])
    max_dev_u = np.array([0.1, 0.1, 0.1, 0.1])  # control bounds
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)

    # Compute LQR solution for terminal cost
    def dlqr(A, B, Q, R, n_steps=500):
        P = Q
        for i in range(n_steps):
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K, P

    K_lqr, P_lqr = dlqr(A, B, Q, R)

    # Setup MPC
    N = 25  # longer horizon for trajectory tracking
    input_data = {
        'rho': 5.0,  # lower initial rho for trajectory tracking
        'A': A,
        'B': B,
        'Q': P_lqr,
        'R': R
    }

    mpc = TinyMPC(input_data, N)

    # Set bounds
    u_max = [1.0-ug[0]] * quad.nu
    u_min = [-ug[0]] * quad.nu
    # x_max = [2.] * quad.nx  # tighter state bounds
    # x_min = [-2.] * quad.nx

    x_max = [5.0] * 3 + [2.0] * 9  # Wider position bounds (first 3 elements)
    x_min = [-5.0] * 3 + [-2.0] * 9
    mpc.set_bounds(u_max, u_min, x_max, x_min)



    # Create trajectory reference
    trajectory = Figure8Reference()
    x_nom = np.zeros((quad.nx, mpc.N))
    u_nom = np.zeros((quad.nu, mpc.N-1))
    for i in range(mpc.N):
        x_nom[:,i] = trajectory.generate_reference(i*quad.dt)
    u_nom[:] = ug.reshape(-1,1)

    # Run simulation
    try:
        print("Starting trajectory tracking simulation...")
        x_all, u_all, iterations = simulate_with_controller(
            x0=x0,
            x_nom=x_nom,
            u_nom=u_nom,
            mpc=mpc,
            quad=quad,
            trajectory=trajectory,
            NSIM=400
        )

        # Compute tracking error
        avg_error, max_error, errors = compute_tracking_error(x_all, trajectory, quad.dt)
        print("\nTracking Error Statistics:")
        print(f"Average L2 Error: {avg_error:.4f} meters")
        print(f"Maximum L2 Error: {max_error:.4f} meters")

        # Save iterations
        np.savetxt('../data/iterations/normal_traj.txt', iterations)

        # Visualize results
        visualize_trajectory(x_all, u_all, trajectory=trajectory, dt=quad.dt)
        plot_iterations(iterations)

        # Plot tracking error
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(errors))*quad.dt, errors)
        plt.xlabel('Time [s]')
        plt.ylabel('L2 Position Error [m]')
        plt.title('Tracking Error over Time')
        plt.grid(True)
        plt.show()

        print("\nSimulation completed successfully!")
        print(f"Average iterations per step: {np.mean(iterations):.2f}")
        print(f"Total iterations: {sum(iterations)}")

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main()