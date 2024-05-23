 """
This script produces the figure for the synthetic inverse problem in
van den Boom and Thiery (2024, arXiv:2404.18556).
"""
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np

import dais


np.random.seed(1)  # Set seed for reproducibility.


## Define the log-likelihood and its gradient
def create_GP_covariance(length_scale = 0.3, 
                         dim_space = 100,
                         eps = 10**-10):
    t = np.linspace(0, 1, dim_space)
    def cov_func(t1, t2, length_scale = 1.0):
        return np.exp(-(t1-t2)**2 / length_scale**2)

    Sigma = np.zeros((dim_space, dim_space))
    for i in range(dim_space):
            Sigma[i,] = cov_func(t[i], t, length_scale)
    
    return t, Sigma + eps*np.eye(dim_space)


length_scale = 0.05
dim_space = 100  # Grid size of function discretization
dim_L = dim_space // 2  # Dimension of output
x_array, Sigma = create_GP_covariance(length_scale, dim_space)
Sigma_inv = np.linalg.inv(Sigma)
n_observations = 30  # Number of observations y_i

index_array_observation = np.random.choice(range(dim_L), 
                                            n_observations, 
                                            replace=True).astype(int)


# Create a "heat-equation"-like blurring kernel.
def compute_heat_matrix(dim, T_horizon = 0.05):
    heat_kernel = np.zeros(dim)
    
    for k in range(dim):
        dist = float( min(k, dim-k) ) / dim
        heat_kernel[k] = np.exp(- (dist / T_horizon)**2 )

    heat_kernel = heat_kernel / np.sum(heat_kernel)
    heat_matrix = np.zeros((dim,dim))

    for k in range(dim):
        heat_matrix[:,k] = np.roll(heat_kernel, k)

    return heat_matrix


T_horizon = 0.05
heat_matrix = compute_heat_matrix(dim_space, T_horizon)


def L_single_F(F, index_array_observation=index_array_observation):
    # Make sure that `index_array_observation` is a Numpy array
    if not hasattr(index_array_observation, "__len__"):
        index_array_observation = np.array([index_array_observation])
    elif isinstance(index_array_observation, range):
        index_array_observation = np.array(index_array_observation)
        
    """
    input
    =====
    F: temperature at time t=0
    
    output
    ======
    temperature at time T=T_horizon on a coarser grid
    """
    G = heat_matrix @ F**3
    return G[:len(G):2][index_array_observation]


def L(F, index_array_observation=index_array_observation):
    
    if len(F.shape) == 1:
        return L_single_F(F, index_array_observation)
    else:
        return np.array([L_single_F(
            F[i,:], index_array_observation
        ) for i in range(F.shape[0])])


def grad_L(F, index_array_observation=index_array_observation):
    """The Jacobian of the map L"""
    return autograd.jacobian(lambda F: L(F, index_array_observation))(F)


# Function that draws F from the prior
def draw_from_prior(Sigma):
    """Draw F from the prior"""
    dim_space, _ = Sigma.shape
    return np.random.multivariate_normal(np.zeros(dim_space), Sigma)


## Generate measurements $y$.
sigma2 = 1.0
gamma = 1.0 / sigma2
f = draw_from_prior(Sigma)

y = np.random.multivariate_normal(
    L(f, index_array_observation), sigma2 * np.identity(n_observations)
)


def log_likelihood(m, sigma2 = sigma2, subset = range(n_observations)):
    """The log-likelihood, up to a constant, evaluated at f = m"""
    return -0.5 * np.sum(
        (y[subset] - L(m, index_array_observation[subset]))**2, axis=m.ndim-1
    ) / sigma2


def log_posterior(m, sigma2 = sigma2, Sigma_inv = Sigma_inv):
    """The log posterior, up to a constant, evaluated at f = m"""
    return -0.5 * m @ Sigma_inv @ m + log_likelihood(m, sigma2)
    

# We use a zero-mean Gaussian prior:
mu_0 = np.zeros(dim_space)  # Prior mean
Sigma_0 = Sigma  # Prior covariance


print("Running MCMC...")
# We run a random walk Metropolis algorithm.
# Basic preconditionned random walk

# Cholesky decomposition for fast sampling from the prior
J = np.linalg.cholesky(Sigma)

mcmc_iter = 10**5
beta = 0.1
m  = draw_from_prior(Sigma)
mcmc_history = np.zeros((mcmc_iter, dim_space))
running_acceptance = 0
log_likelihood_current = log_likelihood(m)

for k in range(mcmc_iter):
    # Generate a proposal that is reversible wrt the prior.
    m_proposal = np.sqrt(1 - beta**2)*m + beta*J@np.random.normal(
        loc=0, scale=1, size=dim_space
    )

    log_likelihood_proposal = log_likelihood(m_proposal)
    log_acceptance =  log_likelihood_proposal - log_likelihood_current
    
    # Metropolis-Hastings accept-reject
    accepted = 0

    if np.log(np.random.rand()) < log_acceptance:
        m = m_proposal
        log_likelihood_current =  log_likelihood_proposal
        accepted = 1
        
    # Monitoring
    running_acceptance = (running_acceptance*k + accepted) / (k+1)

    if k % 10000 == 0:
        print("Iteration = {0} \t Acceptance Rate = {1:2.1f}%".format(
            k, 100 * running_acceptance), end = "\r"
        )

    mcmc_history[k,:] = m

print("Iteration = {0} \t Acceptance Rate = {1:2.1f}%".format(
    k + 1, 100 * running_acceptance)
)

mean_mcmc = np.mean(mcmc_history, axis=0)
percentile_mcmc = np.percentile(mcmc_history, q=[2.5,97.5], axis=0)


print("Computing the Laplace approximation...")


### Define Woodbury functions
def woodbury_solve(Sigma, J, sigma2, b):
    """
    Solve: A*x = b
    with: A = Sigma_inv + J.T@J/sigma2
    """
    d, _ = J.shape
    small_matrix = sigma2*np.eye(d) + J@Sigma@J.T
    return Sigma@b - Sigma@J.T@np.linalg.solve(small_matrix, J @ Sigma @ b)


def woodbury_inverse(Sigma, J, sigma2):
    """
    Compute the inverse of A
    with: A = Sigma_inv + J.T@J/sigma2
    """
    d, _ = J.shape
    small_matrix = sigma2*np.eye(d) + J@Sigma@J.T
    return Sigma - Sigma@J.T@np.linalg.inv(small_matrix)@J@Sigma


def Laplace(Sigma = Sigma, sigma2 = sigma2, verbose = True):
    """
    Laplace approximation

    The algorithm below is equivalent to:
    1. Gauss-Newton algorithm for finding the MAP with dampened updates
    governed by `learning_rate`
    2. Gauss-Newton approximation to the Hessian at the MAP
    """
    # Initialize the mean m of the approximating distribution q.
    m = draw_from_prior(Sigma)
    # We do not initialize at zero as it is a critical point of the likelihood.

    count_iter = -1  # Count number of iterations
    iter_max = 10**4  # Maximum number of iterations
    likelihood_history = np.zeros(iter_max)
    learning_rate = 0.1
    Sigma_inv = np.linalg.inv(Sigma)
    loss = log_posterior(m, sigma2, Sigma_inv)
    non_monotone_linesearch_index = 5

    # The main iterative scheme
    while True:
        count_iter += 1

        if count_iter > iter_max:
            print(
                "Stopping afer", iter_max,
                "iterations. Laplace did not converge."
            )

            break

        J = grad_L(m)  # Compute the Jacobian at the current value of m.

        m_star = woodbury_solve(
            Sigma, J, sigma2,
            J.T @ (y - L(m, index_array_observation) + J@m) / sigma2
        )

        # Compute the descent direction and then adjust the learning rate
        # making sure that the new iterate does not lead to a log-likelihood
        # that is less that the minimum of the last
        # `non_monotone_linesearch_index` iterates.
        descent_direction = m_star - m
        m_new = m + learning_rate * descent_direction
        loss_new = log_posterior(m_new, sigma2, Sigma_inv)

        if (count_iter > non_monotone_linesearch_index):
            threshold = np.min(likelihood_history[
                (count_iter-non_monotone_linesearch_index):(count_iter-1)
            ])

            tmp_count = 0

            while loss_new < threshold and tmp_count < 20:
                learning_rate *= 0.5
                m_new = m + learning_rate*descent_direction
                loss_new = log_posterior(m_new, sigma2, Sigma_inv)
                tmp_count += 1

        loss = loss_new
        delta_mean = np.linalg.norm(learning_rate * descent_direction)
        m = m_new
        learning_rate *= 1.5  # Ramp up a bit the learning rate

        if verbose:
            print("Iteration:{0:3d} \t Delta_mean = {1:5.10f}".format(
                count_iter, delta_mean
            ), end="\r")

        likelihood_history[count_iter] = loss

        if delta_mean < 1e-8:
            if verbose:
                print("Number of Laplace iterations:", count_iter)

            break
    
    if verbose:
        print("Iteration:{0:3d} \t Delta_mean = {1:5.10f}".format(
            count_iter, delta_mean
        ))

        print("Number of Laplace iterations:", count_iter)
    
    Sigma = woodbury_inverse(Sigma, J, sigma2)
    return m, Sigma


m_Laplace, Sigma_Laplace = Laplace()


print("Running EP-IS...")
Lambda_0 = Sigma_inv
gamma = 1/sigma2


# Set up tapering matrix.
def wendland1(h, length_scale):
    x = float(h) / length_scale
    return (max(1-x ,0)**4) * (1+4*x)


def create_wendland_tapering(length_scale, dim_space):
    wendland_taper = np.zeros((dim_space,dim_space))
    for i in range(dim_space):
        for j in range(dim_space):
            h = np.abs(x_array[i] - x_array[j])
            wendland_taper[i,j] = wendland1(h, length_scale)
    return wendland_taper


w_taper = create_wendland_tapering(length_scale = .2, dim_space = dim_space)


def EP_IS_body(m, i, Lambda_min_i, mu_min_i, temper=False, S=10**4):
    if not hasattr(i, "__len__"):
        i = np.array([i])
    
    n = len(i)  # `n` is the rank of the output Lambda_i.
    
    # Generate samples from the proposal.
    L_m = L(m, index_array_observation[i])
    J_m = grad_L(m, index_array_observation[i])
    Lambda_i = gamma * J_m.T @ J_m
    proposal_variance = np.linalg.inv(Lambda_min_i + Lambda_i)
    mu_i = gamma * J_m.T @ (y[i] - L_m + J_m @ m)
    
    if temper:
        return proposal_variance @ (mu_min_i + mu_i), Lambda_i, mu_i
    
    F_prop = np.random.multivariate_normal(
        mean = proposal_variance @ (mu_min_i + mu_i), cov = proposal_variance,
        size = S
    )

    # Compute the importance weights.
    tmp = np.sum((y[i] - L(F_prop, index_array_observation[i]))**2, axis = -1)
    
    # Minimize the number of computations that have to be repeated S times.
    w_log = -0.5 * gamma * (
        tmp - np.sum(((y[i]-L_m+J_m@m) - F_prop@J_m.T)**2, axis = -1)
    )

    w_log -= np.max(w_log)
    w = np.exp(w_log)
    w /= np.sum(w)

    # Importance sampling estimate of the posterior mean
    posterior_mean = w.T @ F_prop
    
    # Importance sampling estimate of the posterior covariance
    tmp = F_prop-posterior_mean
    posterior_covariance = tmp.T @ (tmp * w[:,np.newaxis])
    
    Lambda_i = np.linalg.inv(posterior_covariance * w_taper) - Lambda_min_i
    mu_i = (Lambda_min_i + Lambda_i)@posterior_mean - mu_min_i
    return posterior_mean, Lambda_i, mu_i


def EP_IS(Sigma = Sigma, sigma2 = sigma2, verbose = True):
    """
    EP-IS from van den Boom and Thiery (2019) with covariance matrix tapering
    """
    np.random.seed(19850822)
    m = m_Laplace  # Initialize the mean m of the approximating distribution q.
    EP_iter_max = 30  # Maximum number of EP iterations
    split_size = n_observations // 2
    
    # Number of "likelihood factors" in the EP approximation
    n_splits = n_observations // split_size
    
    splits = np.array(range(0, n_observations)).reshape(n_splits, split_size)
    Lambda = np.zeros((n_splits,dim_space,dim_space))
    mu = np.zeros((n_splits,dim_space))
    delta_mean = np.empty(n_splits)

    # The main EP iterative scheme
    for c_iter in range(EP_iter_max):
        count_iter = c_iter

        if count_iter > EP_iter_max:
            print(
                "Stopping afer", EP_iter_max,
                "iterations. EP did not converge."
            )

            break
        
        
        for i in range(n_splits):
            m_star, Lambda[i,:,:], mu[i,:] = EP_IS_body(
                m, splits[i,:],
                Lambda_min_i = Lambda_0 + np.sum(Lambda, axis=0) \
                    - Lambda[i,:,:],
                # The following line assumes the prior mean equals zero.
                mu_min_i = np.sum(mu, axis=0) - mu[i,:],
                temper = count_iter < 20
            )
            
            descent_direction = m_star - m
            delta_mean[i] = np.linalg.norm(descent_direction)
            m = m_star
        
        delta_m = np.mean(delta_mean)
        
        if verbose:
            print("Iteration:{0:3d} \t Delta_mean = {1:5.10f}".format(
                count_iter, delta_m
            ), end="\r")
    
    if verbose:
        print("Iteration:{0:3d} \t Delta_mean = {1:5.10f}".format(
            count_iter, delta_m
        ))

        print("Number of EP iterations:", count_iter)
    
    return m, np.linalg.inv(Lambda_0 + np.sum(Lambda, axis=0))


m_EP_IS, Sigma_EP_IS = EP_IS()


print("Running DAIS...")

m_DAIS, Sigma_DAIS = dais.DAIS(
    U_scalar=log_posterior, S=10**4, max_iter=100, S_eff_target=10**2,
    µ=m_Laplace, Σ=Sigma
)


# Creating the figure:
def plot_result(ax, m, cov, title):
    marginal_std = np.sqrt(np.diag(cov))
    ax.set_ylim(-2.6, 2.3)
    ax.set_title(title)
    ax.set(xlabel =r"$t$", ylabel=r"$f(t)$")
    ax.plot(x_array, m, "-b", linewidth=2)
    ax.plot(x_array, m + 2*marginal_std, "-b", linewidth=1)
    ax.plot(x_array, m - 2*marginal_std, "-b", linewidth=1)
    ax.plot(x_array, mean_mcmc, ":k", linewidth=2)
    ax.plot(x_array, percentile_mcmc[0,:], ":k", linewidth=1)
    ax.plot(x_array, percentile_mcmc[1,:], ":k", linewidth=1)


fig, axes = plt.subplots(1, 3, figsize = (6, 2), constrained_layout=True)

for i in range(3):
    plot_result(
        ax=axes[i], m=[m_DAIS, m_Laplace, m_EP_IS][i],
        cov=[Sigma_DAIS, Sigma_Laplace, Sigma_EP_IS][i],
        title=["DAIS", "Laplace", "EP-IS"][i]
    )

fig.savefig("inverse_problem.pdf")
