import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy
from ..proba.density import LogDensity

# optax for optimization
import optax


class DoubleStochastic:
    """ Doubly Stochastic Variational Bayes for non-Conjugate Inference

    Reference:
    ----------
    "Doubly Stochastic Variational Bayes for non-Conjugate Inference"  
    Titsias, M. and Lázaro-Gredilla, M., ICML 2014
    """
    def __init__(
                self,
                *,
                logtarget: LogDensity,  # logtarget distribution
                ):
        self.logtarget = logtarget
        self.dim = logtarget.dim

    def run(self,
            *,
            key: jnp.ndarray,                   # random key
            n_samples: int,                     # number of samples
            n_iter: int,                        # number of iterations
            mu_init: jnp.ndarray = None,        # initial location
            log_std_init: jnp.ndarray = None,   # initial log standard deviation
            cov_init: jnp.ndarray = None,       # initial covariance
            cov_chol_init: jnp.ndarray = None,  # initial Cholesky factor of the covariance
            adam_lr: float = 1e-3,              # learning rate for Adam
            verbose: bool = False,              # verbose
            store_params_trace: bool = False,   # store the trace of the parameters
            approx_type: str = "full",          # "diag" or "full" or "mixture"
            mixing_components: int = 10,        # number of components for the mixture
            sticking_the_landing: bool = True,  # whether to stick the landing
            use_jit: bool = True,               # whether to use JIT compilation
            ):
        """
        Run the Doubly Stochastic Variational Bayes algorithm
        
        input:
        =====
        key: jnp.ndarray, random key
        n_samples: int, number of samples
        n_iter: int, number of iterations
        mu_init: jnp.ndarray, initial location
        log_std_init: jnp.ndarray, initial log standard deviation
        cov_init: jnp.ndarray, initial covariance
        cov_chol_init: jnp.ndarray, initial Cholesky factor of the covariance
        adam_lr: float, learning rate for Adam
        verbose: bool, verbose
        store_params_trace: bool, store the trace of the parameters
        approx_type: str, "diag" or "full" or "mixture"
        mixing_components: int, number of components for the mixture
        sticking_the_landing: bool, whether to stick the landing
        use_jit: bool, whether to use JIT compilation
        
        output:
        ======
        output_dict: dict, output dictionary
        """
        
        # check the approx_type is valid
        valid_approx = ["diag", "full", "mixture"]
        error_msg = f"Invalid approx_type. \n Must be one of {valid_approx}"
        assert approx_type in valid_approx, error_msg
        
        ##############################
        # FULL COVARIANCE
        ##############################
        if approx_type == "full":
            # extract initial parameters if provided
            if mu_init is None:
                mu_init = jnp.zeros(self.dim)
            if cov_init is None:
                cov_init = jnp.eye(self.dim)
            if cov_chol_init is None:
                cov_chol_init = jnp.linalg.cholesky(cov_init)

            # initialize the parameters
            mu = mu_init
            cov_chol = cov_chol_init
            cov_chol_diag = jnp.diag(cov_chol)
            cov_chol_lower = jnp.tril(cov_chol, k=-1)
            params = {"mu": mu, "log_diag": jnp.log(cov_chol_diag), "cov_chol_lower": cov_chol_lower}

            def generate_samples(params, key):
                """ generate samples from the variational distribution """
                mu = params["mu"]
                # cov_chol = params["cov_chol"]
                diag = jnp.exp(params["log_diag"])
                cov_chol = jnp.diag(diag) + jnp.tril(params["cov_chol_lower"], k=-1)
                zs = jr.normal(key, shape=(n_samples, self.dim))
                xs = zs @ cov_chol.T + mu[None, :]
                # zs: samples from standard normal
                # xs: samples from q
                return xs, zs
            
            def log_q(params, xs):
                """ log density of the variational distribution """
                mu = params["mu"]
                diag = jnp.exp(params["log_diag"])
                cov_chol = jnp.diag(diag) + jnp.tril(params["cov_chol_lower"], k=-1)
                log_Z = 0.5*self.dim*jnp.log(2*jnp.pi) + 0.5*jnp.sum(jnp.log(diag**2))
                xs_white = jnp.linalg.solve(cov_chol, (xs - mu[None, :]).T).T
                log_q = -0.5*jnp.sum(xs_white**2, axis=1) - log_Z
                return log_q

            def clean_params(params):
                """ post-processing after gradient update """
                cov_chol_lower = params["cov_chol_lower"]
                params["cov_chol_lower"] = jnp.tril(cov_chol_lower, k=-1)
                return params
            
            def postprocess_params(params):
                """ post-processing before returning the parameters to user """
                diag = jnp.exp(params["log_diag"])
                cov_chol = jnp.diag(diag) + jnp.tril(params["cov_chol_lower"], k=-1)
                return {"mu": params["mu"], "cov_chol": cov_chol}
            
            def KL(params, key):
                """ KL divergence between q and p with repametrization trick:
                KL = E_q[log q/p] = -Entropy(q) - E_q[log p]
                q is parametrized by mu and cov_chol
                """
                def compute_entropy(params, xs):
                    """ compute the entropy of the variational distribution
                    (entropy) = -E_q[log q]
                    """
                    # diag = jnp.exp(params["log_diag"])
                    # cst = 0.5*self.dim*jnp.mean(zs**2) + 0.5*self.dim*jnp.log(2*jnp.pi)
                    # entropy = cst + 0.5*jnp.sum(jnp.log(diag**2))
                    entropy = -jnp.mean(log_q(params, xs))
                    return entropy
            
                # generate samples xs from q
                key, key_ = jr.split(key)
                xs, _ = generate_samples(params, key_)
                
                # compute entropy
                if sticking_the_landing:
                    # Reference:
                    # "Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference"
                    # Geoffrey Roeder, Yuhuai Wu, David Duvenaud
                    # https://arxiv.org/abs/1703.09194
                    params_stop = jax.lax.stop_gradient(params)
                    entropy = compute_entropy(params_stop, xs)
                else:
                    entropy = compute_entropy(params, xs)
                entropy = compute_entropy(params_stop, xs)
                kl = -entropy - self.logtarget.batch(xs).mean()
                return kl
            
        ##############################
        # MEAN FIELD
        ##############################
        if approx_type == "diag":
            # extract initial parameters if provided
            if mu_init is None:
                mu_init = jnp.zeros(self.dim)
            if log_std_init is None:
                log_std_init = jnp.zeros(self.dim)

            # initialize the parameters
            params = {"mu": mu_init, "log_diag": log_std_init}

            def generate_samples(params, key):
                """ generate samples from the variational distribution """
                mu = params["mu"]
                stds = jnp.exp(params["log_diag"])
                zs = jr.normal(key, shape=(n_samples, self.dim))
                xs = stds[None, :] * zs + mu[None, :]
                # zs: samples from standard normal
                # xs: samples from q
                return xs, zs
            
            def log_q(params, xs):
                """ log density of the variational distribution """
                mu = params["mu"]
                diag = jnp.exp(params["log_diag"])
                log_Z = 0.5*self.dim*jnp.log(2*jnp.pi) + 0.5*jnp.sum(jnp.log(diag**2))
                xs_white = (xs - mu[None, :]) / diag[None, :]
                log_q = -0.5*jnp.sum(xs_white**2, axis=1) - log_Z
                return log_q

            def clean_params(params):
                return params
            
            def postprocess_params(params):
                """ post-processing before returning the parameters to user """
                log_std = params["log_diag"]
                return {"mu": params["mu"], "log_std": log_std}
            
            def KL(params, key):
                """ KL divergence between q and p with repametrization trick:
                KL = E_q[log q/p] = -Entropy(q) - E_q[log p]
                q is parametrized by mu and cov_chol
                """
                def compute_entropy(params, xs):
                    """ compute the entropy of the variational distribution
                    (entropy) = -E_q[log q]
                    """
                    # diag = jnp.exp(params["log_diag"])
                    # cst = 0.5*self.dim*jnp.mean(zs**2) + 0.5*self.dim*jnp.log(2*jnp.pi)
                    # entropy = cst + 0.5*jnp.sum(jnp.log(diag**2))
                    entropy = -jnp.mean(log_q(params, xs))
                    return entropy
            
                # generate samples xs from q
                key, key_ = jr.split(key)
                xs, zs = generate_samples(params, key_)
                # compute entropy
                if sticking_the_landing:
                    # Reference:
                    # "Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference"
                    # Geoffrey Roeder, Yuhuai Wu, David Duvenaud
                    # https://arxiv.org/abs/1703.09194
                    params_stop = jax.lax.stop_gradient(params)
                    entropy = compute_entropy(params_stop, xs)
                else:
                    entropy = compute_entropy(params, xs)
                entropy = compute_entropy(params_stop, xs)
                kl = -entropy - self.logtarget.batch(xs).mean()
                return kl
            
        ##############################
        # MIXTURE
        ##############################
        if approx_type == "mixture":
            assert mixing_components >= 2, "mixing_components must be >= 2"
            # extract initial parameters if provided
            if mu_init is None:
                mu_init = jnp.zeros(self.dim)
            if log_std_init is None:
                log_std_init = jnp.zeros(self.dim)
            
            # sample K times from N(mu_init, cov_init)
            key, key_ = jr.split(key)
            K = mixing_components
            zs = jr.normal(key_, shape=(K, self.dim))
            # Compute pairwise squared distances
            pairwise_distances_sq = jnp.sum((zs[:, None, :] - zs[None, :, :]) ** 2, axis=-1)
            pairwise_distances = jnp.sqrt(pairwise_distances_sq)
            M = jnp.max(pairwise_distances)
            pairwise_distances = pairwise_distances + jnp.eye(K) * M
            nearest_neighbor_distances = jnp.min(pairwise_distances, axis=1)
            avg_nearest_neighbor_distance = jnp.mean(nearest_neighbor_distances)
            mus = jnp.exp(log_std_init)*zs + mu_init[None, :]
            log_stds = jnp.ones((K, self.dim))*log_std_init + jnp.log(avg_nearest_neighbor_distance / 2.)
            log_alphas = jnp.log(jnp.ones(K)/K)
            
            # initialize the parameters
            params = {"mus": mus, "log_stds": log_stds, "log_alphas": log_alphas}
            
            def generate_samples(p, key):
                """ generate samples from the variational distribution """
                mu = p["mu"]
                stds = jnp.exp(p["log_diag"])
                zs = jr.normal(key, shape=(n_samples, self.dim))
                xs = stds[None, :] * zs + mu[None, :]
                # zs: samples from standard normal
                # xs: samples from q
                return xs, zs
                        
            def log_q(params, xs):
                def log_q_indiv(mu, log_std, xs):
                    """ log density of the variational distribution """
                    diag = jnp.exp(log_std)
                    log_Z = 0.5*self.dim*jnp.log(2*jnp.pi) + 0.5*jnp.sum(jnp.log(diag**2))
                    xs_white = (xs - mu[None, :]) / diag[None, :]
                    return -0.5*jnp.sum(xs_white**2, axis=1) - log_Z
                log_q_indiv_vmap = jax.vmap(log_q_indiv, in_axes=(0, 0, None))
                mus = params["mus"]
                log_stds = params["log_stds"]
                log_alphas = params["log_alphas"]
                alphas = jnp.exp(log_alphas)
                alphas = alphas / jnp.sum(alphas)
                log_alphas = jnp.log(alphas)
                log_qs = log_q_indiv_vmap(mus, log_stds, xs)
                return jax.nn.logsumexp(log_qs + log_alphas[:,None], axis=0)
                
            def clean_params(params):
                log_alphas = params["log_alphas"]
                alphas = jnp.exp(log_alphas)
                alphas = alphas / jnp.sum(alphas)
                params["log_alphas"] = jnp.log(alphas)
                return params
            
            def postprocess_params(params):
                """ post-processing before returning the parameters to user """
                return params

            def KL(params, key):
                """ KL divergence between q and p with repametrization trick:
                KL = E_q[log q/p] = -Entropy(q) - E_q[log p]
                q is parametrized by mu and cov_chol
                """
                # extract mixture parameters
                log_alphas = params["log_alphas"]
                alphas = jnp.exp(log_alphas)
                alphas = alphas / jnp.sum(alphas)
                
                kl_list = []
                for k in range(K):
                    mu = params["mus"][k]
                    log_std = params["log_stds"][k]
                    params_k = {"mu": mu, "log_diag": log_std}
                    
                    # generate samples xs from q_k
                    key, key_ = jr.split(key)
                    xs, _ = generate_samples(params_k, key_)
                    if sticking_the_landing:
                        params_stop = jax.lax.stop_gradient(params)
                        entropy = -jnp.mean(log_q(params_stop, xs))
                    else:
                        entropy = -jnp.mean(log_q(params, xs))
                    kl_list.append(alphas[k] * (-entropy - self.logtarget.batch(xs).mean()))
                # kl = jnp.sum(alphas * kl_arr)
                kl = jnp.sum(jnp.array(kl_list))
                return kl

        # Compute the gradient of the KL divergence
        kl_value_and_grad = jax.value_and_grad(KL)

        # optimizer
        opt = optax.adam(adam_lr)
        opt_state = opt.init(params)

        # update function
        def update(params, opt_state, key):
            kl, grad = kl_value_and_grad(params, key)
            updates, opt_state = opt.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            # clean the parameters
            # eg: ensure triangular part of the Cholesky factor is lower triangular
            params = clean_params(params)
            return params, kl, opt_state

        # JIT compilation
        if use_jit:
            update = jax.jit(update)

        # run the optimization
        kl_init = KL(params, key)
        elbo_trace = [-kl_init]
        params_trace = [postprocess_params(params)]
        
        for it in range(n_iter):
            key, key_ = jr.split(key)
            params, kl, opt_state = update(params, opt_state, key_)
            elbo_trace.append(-kl)
            if store_params_trace:
                params_trace.append(postprocess_params(params))
            if verbose:
                print(f"iter {it}, KL: {kl:.5f}")
        
        output_dict = {
            "params": postprocess_params(params),
            "elbo_trace": elbo_trace,
            "approx_type": approx_type,
        }
        if store_params_trace:
            output_dict["params_trace"] = params_trace
            
        return output_dict
    
    
