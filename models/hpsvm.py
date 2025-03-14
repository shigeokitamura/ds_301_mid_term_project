import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.sparse import diags
import time
import logging
from mpi4py import MPI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("HPSVM")


class HPSVM:
    """
    High-Performance Support Vector Machines implementation as described in the paper.
    This implementation supports both single-node and distributed multi-node computation.
    """

    def __init__(self, tau=1.0, tol=1e-6, max_iter=100, kernel="linear", gamma=0.1):
        """
        Initialize HPSVM.

        Parameters:
        -----------
        tau : float
            Penalty parameter. Higher values give less regularization.
        tol : float
            Tolerance for the stopping criterion.
        max_iter : int
            Maximum number of iterations.
        kernel : str
            Kernel type. Currently supports 'linear' only.
        gamma : float
            Gamma parameter for 'rbf' kernel.
        """
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma

        # Set up MPI environment
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize model parameters
        self.w = None
        self.beta = None
        self.support_indices = None
        self.support_vectors = None
        self.support_vector_labels = None

        # Initialize status flags
        self.is_fitted = False

        # Is this the master node?
        self.is_master = self.rank == 0

        if self.is_master:
            logger.info(f"Initialized HPSVM with {self.size} nodes")
            logger.info(
                f"Parameters: tau={tau}, tol={tol}, max_iter={max_iter}, kernel={kernel}"
            )

    def _initialize_primal_dual_variables(self, n_samples, n_features):
        """Initialize all primal and dual variables."""
        # Initialize variables
        if self.is_master:
            self.w = np.zeros(n_features)
            self.beta = 0.0
        else:
            self.w = None
            # self.beta = None
            self.beta = 0.0

        # Local variables for all nodes
        # Primal variables
        self.z = np.ones(n_samples)

        # Dual variables
        self.v = np.ones(n_samples) * 0.5
        self.s = np.ones(n_samples) * 0.5
        self.u = np.ones(n_samples) * 0.5

        # Initialize diagonal matrices
        self.update_omega()

        # Initialize residuals
        self.r_w = np.zeros(n_features)
        self.rho_beta = 0.0
        self.r_z = np.zeros(n_samples)
        self.r_v = np.zeros(n_samples)
        self.r_u = np.zeros(n_samples)
        self.r_s = np.zeros(n_samples)
        self.r_omega = np.zeros(n_samples)

    def update_omega(self):
        """Update the diagonal matrix Omega."""
        self.omega_diag = (self.z / self.u) + (self.s / self.v)

    def update_residuals(self, X, d, mu):
        """
        Update residuals for the Newton system.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        d : array-like of shape (n_samples,)
            Target labels.
        mu : float
            Barrier parameter.
        """
        Y = X * d.reshape(-1, 1)  # Element-wise multiplication

        # Compute residuals
        if self.w is not None:  # Only on master node
            self.r_w = self.w - Y.T @ self.v
            self.rho_beta = d.T @ self.v

        self.r_z = self.tau * np.ones_like(self.z) - self.v - self.u

        # Compute Y @ w - beta * d - e + z - s
        Yw = X @ self.w if self.w is not None else np.zeros(X.shape[0])
        self.r_v = d * Yw - self.beta * d - np.ones_like(d) + self.z - self.s

        # Compute perturbed residuals for the barrier method
        self.r_u = self.z * self.u - mu * np.ones_like(self.z)
        self.r_s = self.s * self.v - mu * np.ones_like(self.s)

        # Compute combined residual for the reduced system
        r_hat_z = self.r_z + (self.r_u / self.z)
        r_hat_v = self.r_v + (self.r_s / self.v)

        self.r_omega = r_hat_v - ((self.z / self.u) * r_hat_z)

    def compute_newton_step(self, X, d):
        """
        Compute the Newton step for the primal-dual system.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        d : array-like of shape (n_samples,)
            Target labels.

        Returns:
        --------
        delta_w : array-like of shape (n_features,)
            Step direction for w.
        delta_beta : float
            Step direction for beta.
        delta_v : array-like of shape (n_samples,)
            Step direction for v.
        delta_u : array-like of shape (n_samples,)
            Step direction for u.
        delta_s : array-like of shape (n_samples,)
            Step direction for s.
        delta_z : array-like of shape (n_samples,)
            Step direction for z.
        """
        Y = X * d.reshape(-1, 1)  # Element-wise multiplication

        # Compute parts needed for the M matrix
        omega_inv = 1.0 / self.omega_diag
        Y_omega_inv_Y = Y.T @ (omega_inv.reshape(-1, 1) * Y)
        Y_omega_inv_d = Y.T @ (omega_inv * d)
        d_omega_inv_d = d.T @ (omega_inv * d)

        # All-reduce operation to gather matrices on the master node
        Y_omega_inv_Y_global = self.comm.reduce(Y_omega_inv_Y, op=MPI.SUM, root=0)
        Y_omega_inv_d_global = self.comm.reduce(Y_omega_inv_d, op=MPI.SUM, root=0)
        d_omega_inv_d_global = self.comm.reduce(d_omega_inv_d, op=MPI.SUM, root=0)
        sigma = d_omega_inv_d_global

        # Compute residual-related terms
        d_omega_inv_r_omega = d.T @ (omega_inv * self.r_omega)
        Y_omega_inv_r_omega = Y.T @ (omega_inv * self.r_omega)

        # All-reduce operation for residuals
        d_omega_inv_r_omega_global = self.comm.reduce(
            d_omega_inv_r_omega, op=MPI.SUM, root=0
        )
        Y_omega_inv_r_omega_global = self.comm.reduce(
            Y_omega_inv_r_omega, op=MPI.SUM, root=0
        )

        # On master node, solve the system for delta_w
        if self.is_master:
            # Compute rho_beta_hat and r_w_hat
            rho_beta_hat = self.rho_beta - d_omega_inv_r_omega_global
            r_w_hat = self.r_w + Y_omega_inv_r_omega_global

            # Compute M matrix and r_M
            M = (
                np.eye(X.shape[1])
                + Y_omega_inv_Y_global
                - (1.0 / sigma) * np.outer(Y_omega_inv_d_global, Y_omega_inv_d_global)
            )
            r_M = r_w_hat - (1.0 / sigma) * Y_omega_inv_d_global * rho_beta_hat

            # Solve the system M * delta_w = -r_M
            delta_w = solve(M, -r_M)

            # Compute delta_beta
            delta_beta = (1.0 / sigma) * (
                -rho_beta_hat
                + d_omega_inv_d_global * (Y_omega_inv_d_global.T @ delta_w)
            )
        else:
            delta_w = None
            delta_beta = None

        # Broadcast delta_w and delta_beta to all nodes
        delta_w = self.comm.bcast(delta_w, root=0)
        delta_beta = self.comm.bcast(delta_beta, root=0)

        # Compute local delta_v, delta_u, delta_s, delta_z
        delta_v = omega_inv * (-self.r_omega + d * delta_beta - Y @ delta_w)
        delta_z = (self.u / self.z) * (self.r_z + delta_v)
        delta_s = (self.s / self.v) * (self.r_s - delta_v)
        delta_u = (self.u / self.z) * (self.r_u - self.u * delta_z)

        return delta_w, delta_beta, delta_v, delta_u, delta_s, delta_z

    def compute_step_size(self, delta_v, delta_u, delta_s, delta_z):
        """
        Compute step size to maintain positivity of variables.

        Parameters:
        -----------
        delta_v, delta_u, delta_s, delta_z : array-like
            Step directions.

        Returns:
        --------
        alpha : float
            Step size.
        """
        # Compute maximum step size to maintain positivity of variables
        alpha_v = 1.0
        alpha_s = 1.0
        alpha_u = 1.0
        alpha_z = 1.0

        # For v
        neg_indices = delta_v < 0
        if np.any(neg_indices):
            alpha_v = min(alpha_v, np.min(-self.v[neg_indices] / delta_v[neg_indices]))

        # For s
        neg_indices = delta_s < 0
        if np.any(neg_indices):
            alpha_s = min(alpha_s, np.min(-self.s[neg_indices] / delta_s[neg_indices]))

        # For u
        neg_indices = delta_u < 0
        if np.any(neg_indices):
            alpha_u = min(alpha_u, np.min(-self.u[neg_indices] / delta_u[neg_indices]))

        # For z
        neg_indices = delta_z < 0
        if np.any(neg_indices):
            alpha_z = min(alpha_z, np.min(-self.z[neg_indices] / delta_z[neg_indices]))

        # Apply a safety factor
        alpha = 0.95 * min(alpha_v, alpha_s, alpha_u, alpha_z)

        # Collect the minimum step size from all nodes
        alpha = self.comm.allreduce(alpha, op=MPI.MIN)

        return alpha

    def update_variables(
        self, delta_w, delta_beta, delta_v, delta_u, delta_s, delta_z, alpha
    ):
        """
        Update all variables.

        Parameters:
        -----------
        delta_w, delta_beta, delta_v, delta_u, delta_s, delta_z : array-like
            Step directions.
        alpha : float
            Step size.
        """
        if self.is_master:
            self.w += alpha * delta_w
            self.beta += alpha * delta_beta

        self.v += alpha * delta_v
        self.u += alpha * delta_u
        self.s += alpha * delta_s
        self.z += alpha * delta_z

    def compute_duality_gap(self):
        """
        Compute the duality gap.

        Returns:
        --------
        gap : float
            Duality gap.
        """
        local_gap = np.sum(self.z * self.u) + np.sum(self.s * self.v)
        gap = self.comm.allreduce(local_gap, op=MPI.SUM)
        return gap

    def fit(self, X, y):
        """
        Fit the SVM model on the training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Convert y to {-1, 1}
        d = np.where(y > 0, 1, -1)

        n_samples, n_features = X.shape

        # Distribute data to worker nodes
        local_n_samples = n_samples // self.size
        start_idx = self.rank * local_n_samples
        end_idx = (
            start_idx + local_n_samples if self.rank < self.size - 1 else n_samples
        )

        # Get local data
        X_local = X[start_idx:end_idx]
        d_local = d[start_idx:end_idx]
        local_n_samples = X_local.shape[0]

        if self.is_master:
            logger.info(
                f"Starting HPSVM training with {n_samples} samples and {n_features} features"
            )
            logger.info(f"Data distributed among {self.size} nodes")

        # Initialize variables
        self._initialize_primal_dual_variables(local_n_samples, n_features)

        # Interior point method parameters
        mu = 1.0
        mu_factor = 0.1

        # Newton iterations
        for iteration in range(self.max_iter):
            # Update residuals
            self.update_residuals(X_local, d_local, mu)

            # Compute Newton step
            delta_w, delta_beta, delta_v, delta_u, delta_s, delta_z = (
                self.compute_newton_step(X_local, d_local)
            )

            # Compute step size
            alpha = self.compute_step_size(delta_v, delta_u, delta_s, delta_z)

            # Update variables
            self.update_variables(
                delta_w, delta_beta, delta_v, delta_u, delta_s, delta_z, alpha
            )

            # Update omega
            self.update_omega()

            # Check convergence
            gap = self.compute_duality_gap()
            gap_per_sample = gap / n_samples

            if self.is_master and (
                iteration % 5 == 0 or iteration == self.max_iter - 1
            ):
                logger.info(
                    f"Iteration {iteration}: duality gap = {gap_per_sample:.6f}, step size = {alpha:.6f}"
                )

            # Update barrier parameter
            if gap_per_sample < 10 * mu:
                mu = max(mu_factor * mu, self.tol / 10)

            # Check stopping criterion
            if gap_per_sample < self.tol:
                if self.is_master:
                    logger.info(f"Converged after {iteration + 1} iterations")
                break

        # Store support vectors (indices where v is non-zero)
        is_support = self.v > self.tol
        self.support_indices = np.where(is_support)[0] + start_idx
        self.support_vectors = X_local[is_support]
        self.support_vector_labels = d_local[is_support]

        # Gather all support vectors to master node
        all_support_indices = self.comm.gather(self.support_indices, root=0)
        all_support_vectors = self.comm.gather(self.support_vectors, root=0)
        all_support_vector_labels = self.comm.gather(self.support_vector_labels, root=0)

        if self.is_master:
            # Combine all support vectors
            self.support_indices = (
                np.concatenate(all_support_indices)
                if all_support_indices[0].size > 0
                else np.array([])
            )
            self.support_vectors = (
                np.vstack(all_support_vectors)
                if all_support_vectors[0].shape[0] > 0
                else np.array([]).reshape(0, n_features)
            )
            self.support_vector_labels = (
                np.concatenate(all_support_vector_labels)
                if all_support_vector_labels[0].size > 0
                else np.array([])
            )

            logger.info(
                f"Number of support vectors: {len(self.support_indices)} out of {n_samples} samples"
            )

        self.is_fitted = True
        return self

    def decision_function(self, X):
        """
        Compute the decision function for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns:
        --------
        y_score : array-like of shape (n_samples,)
            Decision function values.
        """
        if not self.is_fitted:
            raise ValueError(
                "The model has not been trained yet. Call 'fit' before using this method."
            )

        return X @ self.w

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns:
        --------
        score : float
            Accuracy of the classifier.
        """
        # Convert y to {-1, 1}
        y_true = np.where(y > 0, 1, -1)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)


# Example usage
if __name__ == "__main__":
    # Check if we're running as the master node
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Create a synthetic dataset (only on master node)
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(
            f"Generated dataset with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples"
        )
    else:
        X_train = None
        X_test = None
        y_train = None
        y_test = None

    # Broadcast data to all nodes
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Create and train HPSVM model
    hpsvm = HPSVM(tau=1.0, tol=1e-4, max_iter=50)

    start_time = time.time()
    hpsvm.fit(X_train, y_train)

    if rank == 0:
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")

        # Test the model
        accuracy = hpsvm.score(X_test, y_test)
        print(f"Test accuracy: {accuracy:.4f}")
