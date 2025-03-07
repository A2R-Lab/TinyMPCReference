# src/hybrid_rho_adapter.py
import numpy as np
from utils.hover_simulation import uhover, xg
from autograd import jacobian
import autograd.numpy as anp

class HybridRhoAdapter:
    def __init__(self, rho_base=5.0, rho_min=1.0, rho_max=100.0, tolerance=1.1, method="analytical", clip=False, n_fixed=4, n_adaptive=2):
        # Keep the same interface as RhoAdapter for compatibility
        self.rho_base = rho_base
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.tolerance = tolerance
        self.method = method
        self.clip = clip
        
        # Add hybrid-specific parameters
        self.n_fixed = n_fixed
        self.n_adaptive = n_adaptive
        
        # Fixed rho values - evenly distributed in the range
        self.fixed_rhos = np.linspace(rho_min, rho_max, n_fixed)
        
        # Initialize adaptive rho values (fewer than fixed to respect memory constraint)
        self.adaptive_rhos = np.linspace(rho_min + 10, rho_max - 10, n_adaptive)
        
        # Store derivatives for Taylor updates (computed offline)
        self.derivatives = {}
        
        # Pre-computed matrices for each fixed rho value
        self.fixed_matrices = {}
        
        # Current indices and activity flag
        self.current_fixed_idx = 0
        self.current_adaptive_idx = 0
        self.is_adaptive_active = False
        
        # For tracking performance
        self.rho_history = [rho_base]
        self.residual_history = []
        self.pri_res_history = []
        self.dual_res_history = []
        
        # Counter for adaptation frequency
        self.iteration = 0
        self.adapt_freq = 5
        
        # Track solver iterations
        self.solve_count = 0

    def initialize_derivatives(self, cache, eps=1e-4):
        """Initialize derivatives using autodiff (OFFLINE phase)"""
        #print("Computing LQR sensitivity for hybrid approach")
        
        # This is the OFFLINE phase - we compute derivatives for each fixed rho value
        # These derivatives will be used for Taylor updates in the online phase
        
        def lqr_direct(rho):
            R_rho = cache['R'] + rho * anp.eye(cache['R'].shape[0])
            A, B = cache['A'], cache['B']
            Q = cache['Q']
            
            # Compute P
            P = Q
            for _ in range(10):
                K = anp.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
                P = Q + A.T @ P @ (A - B @ K)
            
            K = anp.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
            C1 = anp.linalg.inv(R_rho + B.T @ P @ B)
            C2 = A - B @ K
            
            return anp.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
        
        # Get dimensions
        m, n = cache['Kinf'].shape
        k_size = m * n
        p_size = n * n
        c1_size = m * m
        c2_size = n * n
        
        # For each fixed rho value AND each adaptive rho value
        # compute derivatives and matrices
        all_rhos = np.concatenate([self.fixed_rhos, self.adaptive_rhos])
        
        for rho in all_rhos:
            # 1. Compute derivatives using autodiff (offline)
            derivs = jacobian(lqr_direct)(rho)
            
            # 2. Store derivatives for this rho value
            self.derivatives[rho] = {
                'dKinf_drho': derivs[:k_size].reshape(m, n),
                'dPinf_drho': derivs[k_size:k_size+p_size].reshape(n, n),
                'dC1_drho': derivs[k_size+p_size:k_size+p_size+c1_size].reshape(m, m),
                'dC2_drho': derivs[k_size+p_size+c1_size:].reshape(n, n)
            }
            
            # 3. For fixed rho values, also pre-compute the matrices
            if rho in self.fixed_rhos:
                # Compute exact matrices for fixed rho values
                R_rho = cache['R'] + rho * np.eye(cache['R'].shape[0])
                Q_rho = cache['Q'] + rho * np.eye(cache['Q'].shape[0])
                
                A, B = cache['A'], cache['B']
                
                # Compute infinite horizon solution
                Kinf = np.zeros(B.T.shape)
                Pinf = np.copy(cache['Q'])
                
                for k in range(5000):
                    Kinf_prev = np.copy(Kinf)
                    Kinf = np.linalg.inv(R_rho + B.T @ Pinf @ B) @ B.T @ Pinf @ A
                    Pinf = Q_rho + A.T @ Pinf @ (A - B @ Kinf)
                    
                    if np.linalg.norm(Kinf - Kinf_prev, 2) < 1e-10:
                        break
                
                AmBKt = (A - B @ Kinf).T
                Quu_inv = np.linalg.inv(R_rho + B.T @ Pinf @ B)
                
                # Store pre-computed matrices
                self.fixed_matrices[rho] = {
                    'Kinf': Kinf,
                    'Pinf': Pinf,
                    'C1': Quu_inv,
                    'C2': AmBKt
                }
        
        # Also compute and store derivatives for the current rho
        current_rho = cache['rho']
        if current_rho not in self.derivatives:
            derivs = jacobian(lqr_direct)(current_rho)
            self.derivatives[current_rho] = {
                'dKinf_drho': derivs[:k_size].reshape(m, n),
                'dPinf_drho': derivs[k_size:k_size+p_size].reshape(n, n),
                'dC1_drho': derivs[k_size+p_size:k_size+p_size+c1_size].reshape(m, m),
                'dC2_drho': derivs[k_size+p_size+c1_size:].reshape(n, n)
            }
        
        # Store derivatives in cache for current rho (initial state)
        cache['dKinf_drho'] = self.derivatives[current_rho]['dKinf_drho']
        cache['dPinf_drho'] = self.derivatives[current_rho]['dPinf_drho']
        cache['dC1_drho'] = self.derivatives[current_rho]['dC1_drho']
        cache['dC2_drho'] = self.derivatives[current_rho]['dC2_drho']

    def initialize_format_matrices(self, nx, nu, N):
        """Pre-allocate matrices for residual computation"""
        # Calculate dimensions
        x_decision_size = nx * N + nu * (N-1)
        constraint_rows = (nx + nu) * (N-1)
        
        # Pre-allocate matrices once
        self.A_matrix = np.zeros((constraint_rows, x_decision_size))
        self.z_vector = np.zeros((constraint_rows, 1))
        self.y_vector = np.zeros((constraint_rows, 1))
        self.x_decision = np.zeros((x_decision_size, 1))
        
        # Pre-compute P matrix structure
        self.P_matrix = np.zeros((x_decision_size, x_decision_size))
        self.q_vector = np.zeros((x_decision_size, 1))
        
        # Store dimensions for reuse
        self.format_nx = nx
        self.format_nu = nu
        self.format_N = N
        
        # For residual computation
        self.Ax_vector = np.zeros_like(self.z_vector)
        self.r_prim_vector = np.zeros_like(self.z_vector)
        self.r_dual_vector = np.zeros_like(self.x_decision)
        self.Px_vector = np.zeros_like(self.x_decision)
        self.ATy_vector = np.zeros_like(self.x_decision)

    def format_matrices(self, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, cache, N):
        """Memory-optimized matrix formatting - same as original RhoAdapter"""
        nx, nu = self.format_nx, self.format_nu
        
        # Fill x_decision in-place
        x_idx = 0
        for i in range(N):
            self.x_decision[x_idx:x_idx+nx, 0] = x_prev[:, i]
            x_idx += nx
            if i < N-1:
                self.x_decision[x_idx:x_idx+nu, 0] = u_prev[:, i]
                x_idx += nu
        
        # Fill A matrix in-place
        A_base, B_base = cache['A'], cache['B']
        
        # Clear A matrix for reuse
        self.A_matrix.fill(0)
        
        # Fill in dynamics and input constraints
        for i in range(N-1):
            # Input constraints
            row_start = i * nu
            col_start = i * (nx+nu) + nx
            self.A_matrix[row_start:row_start+nu, col_start:col_start+nu] = np.eye(nu)
            
            # Dynamics constraints
            row_start = (N-1) * nu + i * nx
            col_start = i * (nx+nu)
            self.A_matrix[row_start:row_start+nx, col_start:col_start+nx] = A_base
            self.A_matrix[row_start:row_start+nx, col_start+nx:col_start+nx+nu] = B_base
            
            next_state_idx = col_start + nx + nu
            if next_state_idx < self.A_matrix.shape[1]:
                self.A_matrix[row_start:row_start+nx, next_state_idx:next_state_idx+nx] = -np.eye(nx)
        
        # Fill z and y vectors in-place
        for i in range(N-1):
            self.z_vector[i*nu:(i+1)*nu, 0] = z_prev[:, i]
            self.z_vector[(N-1)*nu+i*nx:(N-1)*nu+(i+1)*nx, 0] = v_prev[:, i]
            
            self.y_vector[i*nu:(i+1)*nu, 0] = y_prev[:, i]
            self.y_vector[(N-1)*nu+i*nx:(N-1)*nu+(i+1)*nx, 0] = g_prev[:, i]
        
        # Build P matrix (cost matrix)
        Q, R = cache['Q'], cache['R']
        
        # Clear P matrix for reuse
        self.P_matrix.fill(0)
        
        # Fill diagonal blocks
        x_idx = 0
        for i in range(N):
            # State cost
            self.P_matrix[x_idx:x_idx+nx, x_idx:x_idx+nx] = Q
            x_idx += nx
            
            # Input cost
            if i < N-1:
                self.P_matrix[x_idx:x_idx+nu, x_idx:x_idx+nu] = R
                x_idx += nu
        
        # Create q vector (linear cost vector)
        x_idx = 0
        for i in range(N):
            delta_x = x_prev[:, i] - xg[:12]
            self.q_vector[x_idx:x_idx+nx, 0] = Q @ delta_x
            x_idx += nx
            
            if i < N-1:
                delta_u = u_prev[:, i] - uhover
                self.q_vector[x_idx:x_idx+nu, 0] = R @ delta_u
                x_idx += nu
        
        # # Debug prints copied from original
        # print(f"A: {self.A_matrix.shape}")
        # print(f"z: {self.z_vector.shape}")
        # print(f"y: {self.y_vector.shape}")
        # print(f"P: {self.P_matrix.shape}")
        # print(f"q: {self.q_vector.shape}")
        
        return self.x_decision, self.A_matrix, self.z_vector, self.y_vector, self.P_matrix, self.q_vector

    def compute_residuals(self, x, A, z, y, P, q):
        """Memory-optimized residual computation - same as original RhoAdapter"""
        # Pre-allocate vectors for intermediate results if not already done
        if not hasattr(self, 'Ax_vector'):
            self.Ax_vector = np.zeros_like(z)
            self.r_prim_vector = np.zeros_like(z)
            self.r_dual_vector = np.zeros_like(x)
            self.Px_vector = np.zeros_like(x)
            self.ATy_vector = np.zeros_like(x)
        
        # Compute Ax directly into pre-allocated array
        np.matmul(A, x, out=self.Ax_vector)
        
        # Compute primal residual
        np.subtract(self.Ax_vector, z, out=self.r_prim_vector)
        pri_res = np.max(np.abs(self.r_prim_vector))
        pri_norm = max(np.max(np.abs(self.Ax_vector)), np.max(np.abs(z)))
        
        # Compute dual residual components
        np.matmul(P, x, out=self.Px_vector)
        np.matmul(A.T, y, out=self.ATy_vector)
        
        # Compute full dual residual
        self.r_dual_vector = self.Px_vector + q + self.ATy_vector
        dual_res = np.max(np.abs(self.r_dual_vector))
        
        # Compute normalization
        dual_norm = max(np.max(np.abs(self.Px_vector)), 
                    np.max(np.abs(self.ATy_vector)), 
                    np.max(np.abs(q)))
        
        # Store for adaptation decision
        self.pri_res_history.append(pri_res)
        self.dual_res_history.append(dual_res)
        self.residual_history.append((pri_res, dual_res))
        
        return pri_res, dual_res, pri_norm, dual_norm

    def predict_rho(self, pri_res, dual_res, pri_norm, dual_norm, current_rho):
        """Predict new rho value based on residuals"""
        # Increment iteration counter
        self.iteration += 1
        
        # If not adaptation time, return current rho
        if self.iteration % self.adapt_freq != 0:
            return current_rho
        
        # Standard OSQP adaptation formula
        normalized_pri = pri_res / (pri_norm + 1e-10)
        normalized_dual = dual_res / (dual_norm + 1e-10)
        
        ratio = normalized_pri / (normalized_dual + 1e-10)
        ideal_rho = current_rho * np.sqrt(ratio)
        ideal_rho = np.clip(ideal_rho, self.rho_min, self.rho_max)
        
        # If using adaptive set and current rho is in adaptive set
        if self.is_adaptive_active:
            # Update this adaptive rho value
            self.adaptive_rhos[self.current_adaptive_idx] = ideal_rho
            self.rho_history.append(ideal_rho)
            return ideal_rho
        else:
            # For fixed rho, just return the closest value
            new_idx = np.abs(self.fixed_rhos - ideal_rho).argmin()
            self.current_fixed_idx = new_idx
            self.rho_history.append(self.fixed_rhos[new_idx])
            return self.fixed_rhos[new_idx]

    # def select_rho_for_trajectory(self, state, iteration):
    #     """Select appropriate rho based on iteration and recent performance"""
    #     self.solve_count = iteration
        
    #     # Performance-based selection strategy
    #     if len(self.pri_res_history) > 5:
    #         recent_progress = self.pri_res_history[-1] / (self.pri_res_history[-5] + 1e-10)
            
    #         # If convergence is slowing down, switch to adaptive
    #         if recent_progress > 0.7:  # Not converging quickly enough
    #             self.is_adaptive_active = True
    #             # Choose which adaptive rho to use
    #             self.current_adaptive_idx = iteration % self.n_adaptive
    #             return self.adaptive_rhos[self.current_adaptive_idx]
        
    #     # Simple alternating strategy if no performance data yet
    #     if iteration % 10 < 5:
    #         # Use fixed rho set
    #         self.is_adaptive_active = False
    #         self.current_fixed_idx = (iteration // 10) % self.n_fixed
    #         return self.fixed_rhos[self.current_fixed_idx]
    #     else:
    #         # Use adaptive rho set
    #         self.is_adaptive_active = True
    #         self.current_adaptive_idx = (iteration // 10) % self.n_adaptive
    #         return self.adaptive_rhos[self.current_adaptive_idx]

    # def update_matrices(self, cache, new_rho):
    #     old_rho = cache['rho']
    #     delta_rho = new_rho - old_rho
        
    #     updates = {
    #         'rho': new_rho,
    #         'Kinf': cache['Kinf'] + delta_rho * cache['dKinf_drho'],
    #         'Pinf': cache['Pinf'] + delta_rho * cache['dPinf_drho'],
    #         'C1': cache['C1'] + delta_rho * cache['dC1_drho'],
    #        # 'C2': cache['C2'] + delta_rho * cache['dC2_drho']
    #     }


    #     return updates


    def update_matrices(self, cache, new_rho):
        """Update matrices using different strategies based on rho type"""
        old_rho = cache['rho']
        
        # If rho hasn't changed much, skip the update
        if abs(new_rho - old_rho) < 1e-6:
            return {'rho': new_rho}
        
        # STRATEGY 1: For fixed rho values, use pre-computed matrices (fast lookup)
        if new_rho in self.fixed_matrices:
            matrices = self.fixed_matrices[new_rho]
            updates = {
                'rho': new_rho,
                'Kinf': matrices['Kinf'],
                'Pinf': matrices['Pinf'],
                'C1': matrices['C1'],
                'C2': matrices['C2']
            }
            return updates
        
        # STRATEGY 2: For adaptive rho values, use Taylor approximation with bounds
        # Limit rho change per step to avoid numerical issues
        max_step = min(1.0, old_rho * 0.2)  # Smaller step for smaller rho values
        if abs(new_rho - old_rho) > max_step:
            delta_rho = max_step * np.sign(new_rho - old_rho)
            new_rho = old_rho + delta_rho
        else:
            delta_rho = new_rho - old_rho
        
        # Find closest pre-computed derivative
        closest_deriv_rho = min(self.derivatives.keys(), key=lambda x: abs(x - old_rho))
        derivs = self.derivatives[closest_deriv_rho]
        
        # Apply Taylor update with bounds checking
        Kinf_update = cache['Kinf'] + delta_rho * derivs['dKinf_drho']
        Pinf_update = cache['Pinf'] + delta_rho * derivs['dPinf_drho']
        C1_update = cache['C1'] + delta_rho * derivs['dC1_drho']
        C2_update = cache['C2'] + delta_rho * derivs['dC2_drho']
        
        # # Check for extreme values and bound them
        def bound_matrix(matrix, name, max_val=1e3):
            if np.any(np.abs(matrix) > max_val):
                print(f"Warning: Large values in {name}, applying bounds")
                return np.clip(matrix, -max_val, max_val)
            return matrix
        
        # Apply bounds to all updated matrices
        Kinf_update = bound_matrix(Kinf_update, 'Kinf')
        Pinf_update = bound_matrix(Pinf_update, 'Pinf')
        C1_update = bound_matrix(C1_update, 'C1')
        C2_update = bound_matrix(C2_update, 'C2')
        
        updates = {
            'rho': new_rho,
            'Kinf': Kinf_update,
            'Pinf': Pinf_update,
            'C1': C1_update,
            'C2': C2_update
        }
        return updates

    # def update_matrices(self, cache, new_rho):
    #     """Update with fallback mechanism"""
    #     old_rho = cache['rho']
        
    #     try:
    #         # If rho hasn't changed much, skip the update
    #         if abs(new_rho - old_rho) < 1e-6:
    #             return {'rho': new_rho}
            
    #         # STRATEGY 1: For fixed rho values, use pre-computed matrices
    #         if new_rho in self.fixed_matrices:
    #             matrices = self.fixed_matrices[new_rho]
    #             return {
    #                 'rho': new_rho,
    #                 'Kinf': matrices['Kinf'],
    #                 'Pinf': matrices['Pinf'],
    #                 'C1': matrices['C1'],
    #                 'C2': matrices['C2']
    #             }
            
    #         # Limit rho change
    #         max_step = min(1.0, old_rho * 0.2)
    #         if abs(new_rho - old_rho) > max_step:
    #             delta_rho = max_step * np.sign(new_rho - old_rho)
    #             new_rho = old_rho + delta_rho
    #         else:
    #             delta_rho = new_rho - old_rho
            
    #         # Find closest pre-computed derivative
    #         closest_deriv_rho = min(self.derivatives.keys(), key=lambda x: abs(x - old_rho))
    #         derivs = self.derivatives[closest_deriv_rho]
            
    #         # Apply Taylor update
    #         Kinf_update = cache['Kinf'] + delta_rho * derivs['dKinf_drho']
    #         Pinf_update = cache['Pinf'] + delta_rho * derivs['dPinf_drho']
    #         C1_update = cache['C1'] + delta_rho * derivs['dC1_drho']
    #         C2_update = cache['C2'] + delta_rho * derivs['dC2_drho']
            
    #         # Test the updated matrices for numerical stability
    #         # Just check for NaN or Inf values
    #         if (np.any(np.isnan(Kinf_update)) or np.any(np.isinf(Kinf_update)) or
    #             np.any(np.isnan(Pinf_update)) or np.any(np.isinf(Pinf_update)) or
    #             np.any(np.isnan(C1_update)) or np.any(np.isinf(C1_update)) or
    #             np.any(np.isnan(C2_update)) or np.any(np.isinf(C2_update))):
    #             raise RuntimeError("NaN or Inf values detected in Taylor updates")
                
    #         return {
    #             'rho': new_rho,
    #             'Kinf': Kinf_update,
    #             'Pinf': Pinf_update,
    #             'C1': C1_update,
    #             'C2': C2_update
    #         }
            
    #     except Exception as e:
    #         print(f"Taylor update failed: {e}, falling back to unchanged matrices")
    #         # Fallback: only update rho, keep matrices unchanged
    #         return {'rho': new_rho}
        
       
