import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.quadrotor import QuadrotorDynamics
from src.tinympc import TinyMPC
from utils.visualization import visualize_trajectory, plot_iterations, plot_rho_history
from utils.traj_simulation import simulate_with_controller
from scipy.spatial.transform import Rotation as spRot
from utils.reference_trajectories import Figure8Reference
from src.rho_adapter import RhoAdapter
import matplotlib.pyplot as plt

def compute_tracking_error(x_all, trajectory, dt):
    """Compute L2 tracking error over the trajectory"""
    errors = []
    times = np.arange(len(x_all)) * dt
    
    for t, x in zip(times, x_all):
        x_ref = trajectory.generate_reference(t)
        pos_error = np.linalg.norm(x[0:3] - x_ref[0:3])
        errors.append(pos_error)
    
    return np.mean(errors), np.max(errors), errors

def main(use_rho_adaptation=False):
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

    x0 = np.copy(xg)
    
    # Trajectory parameters
    amplitude = 0.5
    w = 2*np.pi/4.0
    trajectory = Figure8Reference(A=amplitude, w=w)
    
    # Get the initial reference point and set initial state
    x_ref_0 = trajectory.generate_reference(0.0)
    x0[0:3] = x_ref_0[0:3]     # Position
    x0[3:7] = qg               # Level quaternion
    x0[7:10] = x_ref_0[6:9]    # All velocities (not just x and z)
    x0[10:13] = np.zeros(3)    # Zero angular velocity

    # Debug print to verify initial state
    print("\nInitial State:")
    print(f"Position: {x0[0:3]}")
    print(f"Quaternion: {x0[3:7]}")
    print(f"Velocity: {x0[7:10]}")
    print(f"Angular velocity: {x0[10:13]}")

    # Cost matrices
    max_dev_x = np.array([
        0.01, 0.01, 0.0001,    # position (tighter)
        0.5, 0.5, 0.05,      # attitude 
        0.5, 0.5, 0.5,       # velocity
        0.7, 0.7, 0.5        # angular velocity
    ])
    max_dev_u = np.array([0.1, 0.1, 0.1, 0.1])

    Q = np.diag(1.0 / (max_dev_x**2))  # Much higher position weights
    R = np.diag(1.0 / (max_dev_u**2))

    # Setup PC
    N = 20
    initial_rho = 1.0
    
    # Initialize MPC
    mpc = TinyMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        Nsteps=N,
        rho=initial_rho
        
    )

   
    if use_rho_adaptation:
        rho_adapter = RhoAdapter(rho_base=initial_rho, rho_min=1.0, rho_max=20.0)
        mpc.rho_adapter = rho_adapter
        mpc.rho_adapter.initialize_derivatives(mpc.cache)
       

    # Set bounds relative to hover thrust
    u_max = np.array([0.3] * quad.nu)  # Allow 30% above hover
    u_min = np.array([-0.3] * quad.nu)  # Allow 30% below hover
    x_max = [2.0] * 3 + [1.0] * 3 + [2.0] * 3 + [2.0] * 3  # [pos, att, vel, omega]
    x_min = [-x for x in x_max]
    mpc.set_bounds(u_max, u_min, x_max, x_min)

    # Create trajectory reference (keep your original parameters)
    x_nom = np.zeros((quad.nx, mpc.N))
    #u_nom = np.zeros((quad.nu, mpc.N-1))
    u_nom = np.zeros((quad.nu, mpc.N-1))

    
    # Initialize with proper timing
    t0 = 0.0
    for i in range(mpc.N):
        t = t0 + i*quad.dt
        x_ref = trajectory.generate_reference(t)
        print(f"Initial trajectory point {i}: {x_ref[0:3]}")
        x_nom[:,i] = x_ref


    u_nom = np.zeros((quad.nu, mpc.N-1))
    u_nom[:] = ug.reshape(-1,1)  # Just use hover thrust

    
    # t0 = 0.0
    # for i in range(mpc.N-1):
    #     t = t0 + i*quad.dt
    #     u_nom[:,i] = trajectory.compute_nominal_control(t, quad)

    # Debug prints
    print("\nNominal control check:")
    print(f"Hover thrust: {quad.hover_thrust}")
    print(f"Initial u_nom: {u_nom[:,0]}")
    print(f"Mid-horizon u_nom: {u_nom[:,mpc.N//2]}")

    # Run simulation
    try:
        print(f"Starting trajectory tracking simulation{'with rho adaptation' if use_rho_adaptation else ''}...")
        simulation_result = simulate_with_controller(
            x0=x0,
            x_nom=x_nom,
            u_nom=u_nom,
            mpc=mpc,
            quad=quad,
            trajectory=trajectory,
            dt_sim=0.01,
            dt_mpc=quad.dt,
            NSIM=250
        )

        # Unpack results based on whether we're using rho adaptation
        if use_rho_adaptation:
            x_all, u_all, iterations, rho_history = simulation_result
        else:
            x_all, u_all, iterations = simulation_result
            rho_history = None

        # Compute and display tracking error
        avg_error, max_error, errors = compute_tracking_error(x_all, trajectory, quad.dt)
        print("\nTracking Error Statistics:")
        print(f"Average L2 Error: {avg_error:.4f} meters")
        print(f"Maximum L2 Error: {max_error:.4f} meters")

        # Save data
        data_dir = Path('../data')
        (data_dir / 'iterations').mkdir(parents=True, exist_ok=True)
        np.savetxt(data_dir / 'iterations' / f"{'adaptive' if use_rho_adaptation else 'normal'}_traj.txt", iterations)
        
        if use_rho_adaptation:
            (data_dir / 'rho_history').mkdir(parents=True, exist_ok=True)
            np.savetxt(data_dir / 'rho_history' / 'adaptive_traj.txt', rho_history)

        # Visualize results
        visualize_trajectory(x_all, u_all, trajectory=trajectory, dt=quad.dt)
        plot_iterations(iterations)
        if use_rho_adaptation:
            plot_rho_history(rho_history)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt', action='store_true', help='Enable rho adaptation')
    args = parser.parse_args()
    
    main(use_rho_adaptation=args.adapt)