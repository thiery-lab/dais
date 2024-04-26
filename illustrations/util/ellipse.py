import numpy as np
import matplotlib


def create_ellipse(mean, covariance, color_ellipse="red", linestyle="-", lw=1):
    """
    Create a 95% Gaussian Ellipse Confidence interval in 2D.

    ref:: http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    ref:: https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    """
    lambda_, v = np.linalg.eig(covariance)
    lambda_ = np.sqrt(lambda_)

    return matplotlib.patches.Ellipse(
        xy=(mean[0], mean[1]),
        width=lambda_[0]*2*np.sqrt(6), height=lambda_[1]*2*np.sqrt(6),
        angle=np.rad2deg(np.arccos(v[0, 0])),
        fill=False,
        color=color_ellipse,
        linewidth=lw, linestyle=linestyle, zorder=5,
        )


def plot_ellipse(ax, µ, Σ, color="black", linestyle="-", lw=1):
    ax.add_artist(create_ellipse(µ, Σ,
                                 color_ellipse=color,
                                 linestyle=linestyle,
                                 lw=lw))
    


