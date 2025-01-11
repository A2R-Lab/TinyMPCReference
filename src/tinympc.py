# src/tinympc.py
import numpy as np

class TinyMPC:
    def __init__(self, A, B, Q, R, Nsteps, rho=1.0, n_dlqr_steps=500, rho_adapter=None, recache = False):
        """Initialize TinyMPC with direct system matrices and compute DLQR automatically
        
        Args:
            A (np.ndarray): System dynamics matrix
            B (np.ndarray): Input matrix
            Q (np.ndarray): State cost matrix
            R (np.ndarray): Input cost matrix
            Nsteps (int): Horizon length
            rho (float): Initial rho value
            n_dlqr_steps (int): Number of steps for DLQR computation
            rho_adapter (object): Rho adapter object
        """
        # Get dimensions
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.N = Nsteps

        # Compute DLQR solution for terminal cost
        P_lqr = self._compute_dlqr(A, B, Q, R, n_dlqr_steps)

        # Initialize cache with computed values
        self.cache = {
            'rho': rho,
            'A': A,
            'B': B,
            'Q': P_lqr,  # Use DLQR solution for terminal cost
            'R': R
        }

        # Initialize state variables
        self.v_prev = np.zeros((self.nx, self.N))
        self.z_prev = np.zeros((self.nu, self.N-1))
        self.g_prev = np.zeros((self.nx, self.N))
        self.y_prev = np.zeros((self.nu, self.N-1))
        self.q_prev = np.zeros((self.nx, self.N))
        
        # Initialize previous solutions for warm start
        self.x_prev = np.zeros((self.nx, self.N))
        self.u_prev = np.zeros((self.nu, self.N-1))

        # Compute cache terms 
        self.compute_cache_terms()

        # Initialize rho adapter AFTER cache is computed
        self.rho_adapter = rho_adapter
        if self.rho_adapter:
            self.rho_adapter.initialize_derivatives(self.cache)
        
        # Set default tolerances and iterations
        self.set_tols_iters()

        self.recache = recache

    def _compute_dlqr(self, A, B, Q, R, n_steps):
        """Compute Discrete-time LQR solution"""
        P = Q
        for _ in range(n_steps):
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return P

    def compute_cache_terms(self):
        """Compute and cache terms for ADMM"""
        Q_rho = self.cache['Q']
        R_rho = self.cache['R']
        R_rho += self.cache['rho'] * np.eye(R_rho.shape[0])
        Q_rho += self.cache['rho'] * np.eye(Q_rho.shape[0])

        A = self.cache['A']
        B = self.cache['B']
        Kinf = np.zeros(B.T.shape)
        Pinf = np.copy(self.cache['Q'])

        # Compute infinite horizon solution
        for k in range(5000):
            Kinf_prev = np.copy(Kinf)
            Kinf = np.linalg.inv(R_rho + B.T @ Pinf @ B) @ B.T @ Pinf @ A
            Pinf = Q_rho + A.T @ Pinf @ (A - B @ Kinf)
            
            if np.linalg.norm(Kinf - Kinf_prev, 2) < 1e-10:
                break

        AmBKt = (A - B @ Kinf).T
        Quu_inv = np.linalg.inv(R_rho + B.T @ Pinf @ B)

        # Cache computed terms
        self.cache['Kinf'] = Kinf
        self.cache['Pinf'] = Pinf
        self.cache['C1'] = Quu_inv
        self.cache['C2'] = AmBKt

    def backward_pass_grad(self, d, p, q, r):
        
        for k in range(self.N-2, -1, -1):

            d[:, k] = np.dot(self.cache['C1'], np.dot(self.cache['B'].T, p[:, k + 1]) + r[:, k])
            p[:, k] = q[:, k] + np.dot(self.cache['C2'], p[:, k + 1]) - np.dot(self.cache['Kinf'].T, r[:, k])
            
           

    def forward_pass(self, x, u, d):
        for k in range(self.N - 1):
            u[:, k] = -np.dot(self.cache['Kinf'], x[:, k]) - d[:, k]
            x[:, k + 1] = np.dot(self.cache['A'], x[:, k]) + np.dot(self.cache['B'], u[:, k])

    def update_primal(self, x, u, d, p, q, r):
        
        
        self.backward_pass_grad(d, p, q, r)
        self.forward_pass(x, u, d)

    def update_slack(self, z, v, y, g, u, x):
        """Update slack variables"""
        for k in range(self.N - 1):
            z[:, k] = np.clip(u[:, k] + y[:, k], self.umin, self.umax)
            v[:, k] = np.clip(x[:, k] + g[:, k], self.xmin, self.xmax)
        v[:, self.N-1] = np.clip(x[:, self.N-1] + g[:, self.N-1], self.xmin, self.xmax)

    def update_dual(self, y, g, u, x, z, v):
        for k in range(self.N - 1):
            y[:, k] += u[:, k] - z[:, k]
            g[:, k] += x[:, k] - v[:, k]
        g[:, self.N-1] += x[:, self.N-1] - v[:, self.N-1]

    def update_linear_cost(self, r, q, p, z, v, y, g, u_ref, x_ref):
        for k in range(self.N - 1):
            r[:, k] = -self.cache['R'] @ u_ref[:, k]
            r[:, k] -= self.cache['rho'] * (z[:, k] - y[:, k])
            
            q[:, k] = -self.cache['Q'] @ x_ref[:, k]
            q[:, k] -= self.cache['rho'] * (v[:, k] - g[:, k])

        p[:,self.N-1] = -np.dot(self.cache['Pinf'], x_ref[:, self.N-1])
        p[:,self.N-1] -= self.cache['rho'] * (v[:, self.N-1] - g[:, self.N-1])

    def set_bounds(self, umax=None, umin=None, xmax=None, xmin=None):
        if (umin is not None) and (umax is not None):
            self.umin = np.array(umin)
            self.umax = np.array(umax)
        if (xmin is not None) and (xmax is not None):
            self.xmin = np.array(xmin)
            self.xmax = np.array(xmax)

    def set_tols_iters(self, max_iter=500, abs_pri_tol=1e-2, abs_dua_tol=1e-2):

        self.max_iter = max_iter
        self.abs_pri_tol = abs_pri_tol
        self.abs_dua_tol = abs_dua_tol

    def update_rho(self):
        """Update rho using the adapter if provided"""
        if self.rho_adapter is None:
            return None

        # Format matrices for residual computation
        x, A, z, y, P, q = self.rho_adapter.format_matrices(
            self.x_prev, self.u_prev, self.v_prev, self.z_prev,
            self.g_prev, self.y_prev, self.cache, self.N
        )
        
        # Compute residuals
        pri_res, dual_res, pri_norm, dual_norm = self.rho_adapter.compute_residuals(
            x, A, z, y, P, q
        )
        
        # Update rho using pre-computed derivatives
        new_rho = self.rho_adapter.predict_rho(
            pri_res, dual_res, pri_norm, dual_norm, self.cache['rho']
        )
        updates = self.rho_adapter.update_matrices(self.cache, new_rho)
        self.cache.update(updates)

        
        
        return new_rho

    def solve_admm(self, x_init, u_init, x_ref=None, u_ref=None):

        if not hasattr(self, 'solve_count'):
            self.solve_count = 0

        self.solve_count += 1

        status = 0
        x = np.copy(x_init)
        u = np.copy(u_init)


        v = np.copy(self.v_prev)
        z = np.copy(self.z_prev)
        g = np.copy(self.g_prev)
        y = np.copy(self.y_prev)
        q = np.copy(self.q_prev)

        # Keep track of previous values for residuals
        v_prev = np.copy(v)
        z_prev = np.copy(z)
    
        r = np.zeros(u.shape)
        p = np.zeros(x.shape)
        d = np.zeros(u.shape)

        if (x_ref is None):
            x_ref = np.zeros(x.shape)
        if (u_ref is None):
            u_ref = np.zeros(u.shape)


        #if trajectory following, set max_iter to 10
        self.max_iter = 20

        for k in range(self.max_iter):
            
    
            self.update_primal(x, u, d, p, q, r)
            self.update_slack(z, v, y, g, u, x)
            self.update_dual(y, g, u, x, z, v)
            self.update_linear_cost(r, q, p, z, v, y, g, u_ref, x_ref)

            pri_res_input = np.max(np.abs(u - z))
            pri_res_state = np.max(np.abs(x - v))
            dua_res_input = np.max(np.abs(self.cache['rho'] * (z_prev - z)))
            dua_res_state = np.max(np.abs(self.cache['rho'] * (v_prev - v)))


            # if self.solve_count < 2:
            #     print(f"Solve {self.solve_count + 1} - Iteration {k}")
            #     print(f"Pri Res Input: {pri_res_input}")
            #     print(f"Dua Res Input: {dua_res_input}")
            #     print(f"Pri Res State: {pri_res_state}")
            #     print(f"Dua Res State: {dua_res_state}")



            z_prev = np.copy(z)
            v_prev = np.copy(v)

            

            if (pri_res_input < self.abs_pri_tol and dua_res_input < self.abs_dua_tol and
                pri_res_state < self.abs_pri_tol and dua_res_state < self.abs_dua_tol):
                print("Converged after ", k, " iterations")
                status = 1
                break


        self.x_prev = x
        self.u_prev = u
        self.v_prev = v
        self.z_prev = z
        self.g_prev = g
        self.y_prev = y
        self.q_prev = q

        if self.rho_adapter is not None:
                self.update_rho()

        if self.recache:
            print("Recaching cache terms")
            self.compute_cache_terms()


        return x, u, status, k

    
    