"""
This script produces the figure for the two-dimensional synthetic examples in
van den Boom, Cremaschi and Thiery (2024, arXiv:2404.18556).
"""
import jax
import jax.numpy as np
import numpy as onp
import matplotlib.patches
import matplotlib.pyplot as plt
import scipy.integrate

import dais


onp.seterr(divide="ignore")  # Suppress np.log(0.0) warning.


def integrate(func):
    return scipy.integrate.nquad(
        func=func,
        ranges=[[-onp.inf, onp.inf], [-onp.inf, onp.inf]],
        opts={"limit": 200}
    )[0]


def create_ellipse(mean, covariance, color_ellipse="red", linestyle="-"):
    """
    Create a 95% Gaussian Ellipse Confidence interval in 2D.
    
    ref:: http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    ref:: https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    """
    lambda_, v = onp.linalg.eig(covariance)
    lambda_ = onp.sqrt(lambda_)
    
    return matplotlib.patches.Ellipse(
        xy=(mean[0], mean[1]),
        width=lambda_[0]*2*np.sqrt(6), height=lambda_[1]*2*np.sqrt(6),
        angle=np.rad2deg(np.arccos(v[0, 0])),
        fill=False,
        color = color_ellipse,
        linewidth=2, linestyle=linestyle, zorder=5
    )


def plot_ellipse(µ, Σ, color="black", label="", linestyle="-"):
    ax.scatter(µ[0], µ[1], c=color, s=20, label=label, zorder=10)

    ax.add_artist(create_ellipse(
        µ, Σ, color_ellipse=color, linestyle=linestyle
    ))


plots = [plt.subplots(
    1, 2, figsize=3.5 * np.array([2, 1]), constrained_layout=True
) for _ in range(2)]

d = 2  # The dimensionality of x


for ind in range(2):
    ## Define the log posterior density and its gradient.
    if ind == 0:
        print("Banana-shaped synthetic distribution")
        Σ = np.array([[1, 0.9], [0.9, 1]])
        Σ_inv = np.linalg.inv(Σ)
        Σ_inv_onp = onp.asarray(Σ_inv)


        def U_scalar(x):
            """
            The log posterior, up to a constant
            
            This is the banana-shaped synthetic distribution from Table 1 of
            Ruiz and Titsias (2019).
            """
            m = np.array([x[0], x[1] + x[0]*x[0] + 1])
            return -0.5 * m @ Σ_inv @ m


        def U_scalar_onp(x):
            """
            The log posterior, up to a constant
            
            This is the banana-shaped synthetic distribution from Table 1 of
            Ruiz and Titsias (2019, http://proceedings.mlr.press/v97/ruiz19a.html).

            This function uses numpy instead of jax.numpy as it results in
            faster numerical integration.
            """
            m = onp.array([x[0], x[1] + x[0]*x[0] + 1])
            return -0.5 * m @ Σ_inv_onp @ m


    else:  # Synthetic mixture distribution
        print("Synthetic mixture distribution")
        weight = np.array([0.3, 0.8])
        mean = np.array([0.8, -2.0])
        n_cmpnt = len(weight)

        Σ_onp = onp.array([
            [[1.0, 0.8], [0.8, 1.0]],
            [[1.0, -0.6], [-0.6, 1.0]],
        ])

        Σ_inv_onp = onp.empty((n_cmpnt, d, d))

        for i in range(n_cmpnt):
            Σ_inv_onp[i, :, :] = onp.linalg.inv(Σ_onp[i, :, :])

        Σ = np.asarray(Σ_onp)
        Σ_inv = np.asarray(Σ_inv_onp)

        weight_det = weight / np.sqrt(np.array(
            [np.linalg.det(Σ[i, :, :]) for i in range(n_cmpnt)]
        ))

        weight_det_onp = onp.asarray(weight_det)


        def U_scalar(x):
            """
            The log posterior, up to a constant
            
            This is the synthetic mixture distribution from Table 1 of
            Ruiz and Titsias (2019, http://proceedings.mlr.press/v97/ruiz19a.html).
            """
            m = np.array([x - mean[i] for i in range(n_cmpnt)])
            
            return np.log(np.sum(weight_det * np.array([
                np.exp(-0.5 * m[i, :] @ Σ_inv[i, :, :] @ m[i, :])
                for i in range(n_cmpnt)
            ])))


        def U_scalar_onp(x):
            """
            The log posterior, up to a constant
            
            This is the synthetic mixture distribution from Table 1 of
            Ruiz and Titsias (2019, http://proceedings.mlr.press/v97/ruiz19a.html).

            This function uses numpy instead of jax.numpy as it results in
            faster numerical integration.
            """
            m_T = onp.subtract.outer(x, mean)
            
            return onp.log(onp.sum(weight_det_onp * onp.array([
                onp.exp(-0.5 * m_T[:, i] @ Σ_inv_onp[i, :, :] @ m_T[:, i])
                for i in range(n_cmpnt)
            ])))


    # We do not initialize with the identity covariance for the
    # banana-shaped example as that is a too good importance sampling
    # proposal distribution resulting in DAIS uninterestingly finishing in
    # one interation.
    Σ_init = np.identity(d) if ind == 1 else Σ
    
    print("Running DAIS...")

    µ_DAIS, Σ_DAIS = dais.DAIS(
        U_scalar=U_scalar, S=10**5, S_eff_target=10**3, Σ=Σ_init
    )

    print("Running DAIS with lower sample size...")

    µ_DAIS_ε, Σ_DAIS_ε = dais.DAIS(
        U_scalar=U_scalar, S=10**3 + 10, max_iter = 400, S_eff_target=10**3,
        Σ=Σ_init
    )

    print("Computing the mean and covariance using numerical integration...")
    
    normalization_constant = integrate(
        lambda x0, x1: onp.exp(U_scalar_onp([x0, x1]))
    )


    def π(x):
        return onp.exp(U_scalar_onp(x)) / normalization_constant


    µ_exact = onp.empty(d)
    µ_exact[0] = integrate(lambda x0, x1: x0 * π([x0, x1]))
    µ_exact[1] = integrate(lambda x0, x1: x1 * π([x0, x1]))

    Σ_exact = onp.empty((d, d))

    Σ_exact[0, 0] = integrate(
        lambda x0, x1: (x0 - µ_exact[0]) * (x0 - µ_exact[0]) * π([x0, x1])
    )

    Σ_exact[0, 1] = integrate(
        lambda x0, x1: (x0 - µ_exact[0]) * (x1 - µ_exact[1]) * π([x0, x1])
    )

    Σ_exact[1, 0] = Σ_exact[0, 1]

    Σ_exact[1, 1] = integrate(
        lambda x0, x1: (x1 - µ_exact[1]) * (x1 - µ_exact[1]) * π([x0, x1])
    )

    print("Running variational inference...")
    # Black-box VI in Python based on
    # https://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf
    U = jax.jit(jax.vmap(U_scalar))


    # We parametrize Σ by its marginal variances and covariance such that x is
    # 5-dimensional.
    def flatten(µ, Σ):
        return np.concatenate((µ, np.diag(Σ), Σ[0, 1, np.newaxis]))


    def unflatten(x):
        Σ = np.diag(x[2:4])
        return x[:2], Σ.at[[[0, 1], [1, 0]]].set(x[4])


    def variational_objective(key, variational_params):
        """
        Stochastic estimate of the variational objective
        
        The variational objective is D(Q || Π), the
        Kullback-Leibler divergence from the target distribution
        Π to the variational approximation Q = N(µ, Σ).
        Π has as log density U plus some constant.
        """
        µ, Σ = unflatten(variational_params)

        # We drop additive constants from the variational objective.
        return -0.5*np.linalg.slogdet(Σ)[1] - np.mean(U(
            jax.random.multivariate_normal(key=key, mean=µ, cov=Σ, shape=(1000,))
        ))


    def adam(
        grad, µ, Σ, num_iters=100, step_size=0.001, b1=0.9, b2=0.999,
        eps=10**-8, plot_trace=False, objective=None,
        rng=jax.random.PRNGKey(1)
    ):
        """
        Adam as described in http://arxiv.org/pdf/1412.6980.pdf
        
        It's basically RMSprop with momentum and some correction terms.
        This function is derived from the function `adam` in
        https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py.
        """
        if plot_trace and objective is None:
            raise ValueError(
                "The argument `objective` must be defined if `plot_trace` is "
                    + "`True`."
            )
        
        x = flatten(µ, Σ)
        m = onp.zeros(len(x))
        v = onp.zeros(len(x))
        trace = onp.zeros(num_iters)
        
        for i in range(num_iters):
            rng, rng_input = jax.random.split(rng)
            
            if plot_trace:
                trace[i] = objective(rng_input, x)
            
            g = np.maximum(np.minimum(grad(rng_input, x), 1e5), -1e5)
            m = (1 - b1) * g      + b1 * m  # First  moment estimate
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate
            mhat = m / (1 - b1**(i + 1))    # Bias correction
            vhat = v / (1 - b2**(i + 1))
            x = x - step_size*mhat/(onp.sqrt(vhat) + eps)
            
            # Ensure Σ's positive definiteness
            x = x.at[2:3].set(max(x[2:3], 0))
            tmp = 0.99 * np.sqrt(x[2] * x[3])
            x = x.at[4].set(max(min(x[4], tmp), -tmp))
        
        if plot_trace:
            plt.plot(trace)
        
        return unflatten(x)


    µ_VI, Σ_VI = adam(
        grad=jax.jit(jax.grad(fun=variational_objective, argnums=1)),
        µ=np.zeros(d), Σ=Σ_init, num_iters=200, step_size=0.01,
        objective=jax.jit(variational_objective)
    )


    # Plot the target distribution.
    # Grid of 2D points
    x_grid = onp.array(np.meshgrid(onp.linspace(
        start=-4.0 if ind == 0 else -6.0, stop=4.0 if ind == 0 else 3.0,
        num=200
    ), onp.linspace(start=-6.5 if ind == 0 else -6.0, stop=3.0, num=200)))
    
    π_grid = onp.empty(x_grid.shape[1:3])

    for i in range(x_grid.shape[1]):
        for j in range(x_grid.shape[2]):
            π_grid[i,j] = π(x_grid[:,i,j])


    for ind2 in range(2):  # Create separate figure for µ_DAIS_ε.
        ax = plots[ind2][1][ind]

        ax.imshow(
            X=-π_grid,  # The negative sign inverts the colors.
            cmap="gray",
            origin="lower",
            extent=[
                x_grid[0,:,:].min(), x_grid[0,:,:].max(), x_grid[1,:,:].min(),
                x_grid[1,:,:].max()
            ]
        )

        ax.set_title(["Banana", "Mixture"][ind])
        ax.grid(True)
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")

        plot_ellipse(µ_VI, Σ_VI, color="red", label="VI", linestyle="--")

        plot_ellipse(
            [µ_DAIS, µ_DAIS_ε][ind2], [Σ_DAIS, Σ_DAIS_ε][ind2], color="blue",
            label="DAIS", linestyle="-"
        )

        plot_ellipse(
            µ_exact, Σ_exact, color="black", label="Exact", linestyle=":"
        )

        if ind == 0:
            ax.legend(loc="upper left")


for ind2 in range(2):
    plots[ind2][0].savefig(
        "banana_and_mixture" + ("_eps" if ind2 else "") + ".pdf"
    )
