import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy
from dais.proba.density import LogDensity

# optax for optimization
import optax


class OAIS:
    """ Optimized Adaptive Importance Sampling 
    
    Reference:
    ----------
    Akyildiz, O. D. and J. Míguez (2021). Convergence rates for optimised
    adaptive importance samplers. Statistics and Computing 31 (2), 12.
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
            approx_type: str = "full",          # "diag" or "full"
            use_jit: bool = True,               # whether to use JIT compilation
            ):
        # check the approx_type is valid
        valid_approx = ["diag", "full"]
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
            params = {"mu": mu,
                    "log_diag": jnp.log(cov_chol_diag),
                    "cov_chol_lower": cov_chol_lower,
                    }

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
                return {"mu": params["mu"],
                        "cov_chol": cov_chol,
                        }
            
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
            
        def objective(params, key):
            """
            R(param) = E_q[pi^2(X) / q^2(X)]
            """
            # generate samples xs from q
            key, key_ = jr.split(key)
            xs, _ = generate_samples(params, key_)
            
            # compute the importance weights
            log_pi = self.logtarget.batch(xs)
            log_q_ = log_q(params, xs)
            log_w = log_pi - log_q_
            weights = jnp.exp(log_w)
            obj = jnp.mean(weights**2)
            return obj
        
        def objective_no_reparam(params, key):
            """
            to compute the gradient
            \nabla R(param) = -E[pi^2(X) / q^2(X) * \nabla log q(X)] where X ~ q
            which is the gradient of:
            -E[ stop_gradient[pi^2(Y) / q^2(Y)] * \nabla log q(Y)]
            where Y = stop_gradient(X) and X ~ q
            """
            # generate samples xs from q
            key, key_ = jr.split(key)
            xs, _ = generate_samples(params, key_)
            
            # compute the importance weights
            log_pi = self.logtarget.batch(xs)
            log_q_ = log_q(params, jax.lax.stop_gradient(xs))
            log_w = log_pi - log_q_
            weights = jnp.exp(log_w)
            obj = jnp.mean(jax.lax.stop_gradient(weights**2) * (0 - log_q_))
            return obj

        # Compute the gradient of the KL divergence
        # objective_grad = jax.grad(objective)
        objective_grad = jax.grad(objective_no_reparam)

        # optimizer
        opt = optax.adam(adam_lr)
        opt_state = opt.init(params)

        # update function
        def update(params, opt_state, key):
            obj = objective(params, key)
            grad = objective_grad(params, key)
            updates, opt_state = opt.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            # clean the parameters
            # eg: ensure triangular part of the Cholesky factor is lower triangular
            params = clean_params(params)
            return params, obj, opt_state

        # JIT compilation
        if use_jit:
            update = jax.jit(update)

        # run the optimization
        obj_init = objective(params, key)
        print(f"Initial Objective: {obj_init:.5f}")
        obj_trace = [obj_init]
        params_trace = [postprocess_params(params)]
        
        for it in range(n_iter):
            key, key_ = jr.split(key)
            params, obj, opt_state = update(params, opt_state, key_)
            obj_trace.append(obj)
            if store_params_trace:
                params_trace.append(postprocess_params(params))
            if verbose:
                print(f"iter {it}, Objective: {obj:.5f}")
        
        output_dict = {
            "params": postprocess_params(params),
            "objective_trace": obj_trace,
            "approx_type": approx_type,
        }
        if store_params_trace:
            output_dict["params_trace"] = params_trace
            
        return output_dict
    
    
