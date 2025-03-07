# src/rho_adapter.py
import autograd.numpy as np
from scipy.linalg import block_diag
from utils.hover_simulation import uhover, xg
from autograd import jacobian

class RhoAdapter:
    def __init__(self, rho_base=85.0, rho_min=60.0, rho_max=100.0, tolerance=1.1, method="analytical", clip = False):
        self.rho_base = rho_base
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.tolerance = tolerance
        self.method = method  # "analytical" or "heuristic"
        self.rho_history = [rho_base]
        self.residual_history = []
        self.derivatives = None
        self.clip = clip

    def initialize_derivatives(self, cache, eps=1e-4):
        """Initialize derivatives using autodiff"""
        #print("Computing LQR sensitivity")
        
        def lqr_direct(rho):
            R_rho = cache['R'] + rho * np.eye(cache['R'].shape[0])
            A, B = cache['A'], cache['B']
            Q = cache['Q']
            
            # Compute Pgit ad
            P = Q
            for _ in range(100):
                K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
                P = Q + A.T @ P @ (A - B @ K)
            
            K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
            C1 = np.linalg.inv(R_rho + B.T @ P @ B)
            C2 = A - B @ K
            
            return np.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
        
        # Get derivatives using autodiff
        m, n = cache['Kinf'].shape
        derivs = jacobian(lqr_direct)(cache['rho'])
        
        # Reshape derivatives into matrices and store in cache
        k_size = m * n
        p_size = n * n
        c1_size = m * m
        c2_size = n * n
        
        # Store derivatives in the cache directly
        cache['dKinf_drho'] = derivs[:k_size].reshape(m, n)
        cache['dPinf_drho'] = derivs[k_size:k_size+p_size].reshape(n, n)
        cache['dC1_drho'] = derivs[k_size+p_size:k_size+p_size+c1_size].reshape(m, m)
        cache['dC2_drho'] = derivs[k_size+p_size+c1_size:].reshape(n, n)

   

    def initialize_format_matrices(self, nx, nu, N):
        """Pre-allocate matrices during initialization to avoid repeated allocation"""
        # Calculate dimensions
        x_decision_size = nx * N + nu * (N-1)
        constraint_rows = (nx + nu) * (N-1)
        
        # Pre-allocate matrices once
        self.A_matrix = np.zeros((constraint_rows, x_decision_size))
        self.z_vector = np.zeros((constraint_rows, 1))
        self.y_vector = np.zeros((constraint_rows, 1))
        self.x_decision = np.zeros((x_decision_size, 1))
        
        # Pre-compute P matrix structure (will be filled with actual values later)
        self.P_matrix = np.zeros((x_decision_size, x_decision_size))
        self.q_vector = np.zeros((x_decision_size, 1))
        
        # Store dimensions for reuse
        self.format_nx = nx
        self.format_nu = nu
        self.format_N = N

    def format_matrices(self, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, cache, N):
        """Memory-optimized matrix formatting"""

        if not hasattr(self, 'format_nx'):
            self.initialize_format_matrices(x_prev.shape[0], u_prev.shape[0], N)

        nx, nu = self.format_nx, self.format_nu
        
        # Fill x_decision in-place
        x_idx = 0
        for i in range(N):
            self.x_decision[x_idx:x_idx+nx, 0] = x_prev[:, i]
            x_idx += nx
            if i < N-1:
                self.x_decision[x_idx:x_idx+nu, 0] = u_prev[:, i]
                x_idx += nu
        
        # Fill A matrix in-place (reuse without recreating)
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
        
        # Build P matrix (cost matrix) - reuse same structure
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
        
        # print(f"A: {self.A_matrix.shape}")
        # print(f"z: {self.z_vector.shape}")
        # print(f"y: {self.y_vector.shape}")
        # print(f"P: {self.P_matrix.shape}")
        # print(f"q: {self.q_vector.shape}")
        
        return self.x_decision, self.A_matrix, self.z_vector, self.y_vector, self.P_matrix, self.q_vector

    def compute_residuals(self, x, A, z, y, P, q):
        """Memory-optimized residual computation"""
        # Pre-allocate vectors for intermediate results
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
        
        return pri_res, dual_res, pri_norm, dual_norm

    def predict_rho(self, pri_res, dual_res, pri_norm, dual_norm, current_rho):
        """Predict new rho value based on residuals"""

        normalized_pri = pri_res / (pri_norm + 1e-10)
        normalized_dual = dual_res / (dual_norm + 1e-10)

        ratio = normalized_pri / (normalized_dual + 1e-10)
            
        rho_new = current_rho * np.sqrt(ratio)
            

        rho_new = np.clip(rho_new, self.rho_min, self.rho_max)

        return rho_new


        # if rho_new >= 1.2*current_rho or rho_new <= current_rho/1.2:
        #     return rho_new


        self.rho_history.append(rho_new)
        return rho_new



    def update_matrices(self, cache, new_rho):
        """Update matrices using derivatives stored in cache"""
        #("Updating matrices")
        old_rho = cache['rho']
        delta_rho = new_rho - old_rho

        if 'dKinf_drho' not in cache:
            self.initialize_derivatives(cache)
        
        updates = {
            'rho': new_rho,
            'Kinf': cache['Kinf'] + delta_rho * cache['dKinf_drho'],
            'Pinf': cache['Pinf'] + delta_rho * cache['dPinf_drho'],
            'C1': cache['C1'] + delta_rho * cache['dC1_drho'],
           'C2': cache['C2'] + delta_rho * cache['dC2_drho']
           #'C2': cache['C2']
        }

        # #return same cache
        # updates = {'rho': new_rho,
        # 'Kinf': cache['Kinf'] + delta_rho * cache['dKinf_drho'],
        # 'Pinf': cache['Pinf'] + delta_rho * cache['dPinf_drho'],
        # 'C1': cache['C1'] + delta_rho * cache['dC1_drho'],
        # 'C2': cache['C2']
        # }
        
        return updates

