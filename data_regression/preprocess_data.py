####################################################################################################
# This script downloads the datasets used for the analysis of the paper
# and transforms the continuous covariates with the minimum description length (MDL) method
# of Fayyad and Irani (1993).
# The datasets are then saved as `.npz` files.
####################################################################################################
import urllib.request
import IPython.core.magics.execution as IPy_exec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages


# Load the R package for discretization.
utils = rpackages.importr("utils")

if not rpackages.isinstalled("discretization"):
    utils.install_packages(
        "discretization", repos="https://cloud.r-project.org"
    )

discretization = rpackages.importr("discretization")
rpy2.robjects.numpy2ri.activate() # Enable passing NumPy arrays to R.


# dataset to load
data_sets = dict(
    spam=dict(url="spambase/spambase"),
    krkp=dict(url="chess/king-rook-vs-king-pawn/kr-vs-kp"),
    ionosphere=dict(url="ionosphere/ionosphere"),
    mushroom=dict(url="mushroom/agaricus-lepiota"),
)


for data_name in data_sets.keys():
    print("\nProcessing the", data_name, "data...")
    # Load the data set
    print("\t Loading the data set...")
    url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    url = url_base + data_sets[data_name]["url"] + ".data"
    tmp = np.genfromtxt(urllib.request.urlopen(url), dtype=str, delimiter=",")
    print("\t The data set has", tmp.shape[0], "rows and", tmp.shape[1], "columns.")
    
    # preprocess the data
    print("\t Discretizing the data...")
    if data_name == "spam":
        y = tmp[:, -1] == "0"
        tmp = tmp[:, :-1].astype(float)
        tmp = np.asarray(discretization.mdlp(np.hstack((tmp, y[:, np.newaxis])))[1])[:, :-1].astype(int)
    elif data_name == "krkp":
        y = tmp[:, -1] == "won"
        tmp = tmp[:, :-1]
    elif data_name == "ionosphere":
        y = tmp[:, -1] == "g"
        tmp = tmp[:, :-1].astype(float)
        tmp = np.hstack((
            tmp[:, :2], # The first two columns are already discretized.
            np.asarray(discretization.mdlp(np.hstack((tmp[:, 2:], y[:, np.newaxis])))[1])[:, :-1])).astype(int)
    elif data_name == "mushroom":
        y = tmp[:, 0] == "e"
        tmp = tmp[:, 1:]
    else:
        raise ValueError("Unknown data set.")
    
    print("\t Discretization completed.")
    
    # Transform y to a jax.numpy.array of -1s and 1s.
    y = np.asarray(2*y - 1)
    dim = tmp.shape[1]
    
    # Ensure that the first category is the most common for dummy variable
    # encoding.
    for j in range(dim):
        counts = np.unique(tmp[:, j], return_counts=True)
        tmp[tmp[:, j] == counts[0][np.argmax(counts[1])], j] = 0
    
    dummies = [pd.get_dummies(tmp[:, j], drop_first=True).values for j in range(dim)]
    X = np.hstack(dummies)

    # add intercept to X
    n = len(y)
    X = np.hstack((np.ones((n, 1)), X))


    dim_final = X.shape[1]
    print("\t Number of covariates:", dim_final)
    
    # create a dictionary with the data
    regression_data = dict(X=X, y=y)
    
    # save the data to disk in numpy format
    save_file = data_name + ".npz"
    # save the data
    np.savez(save_file, **regression_data)


# sanity check: load the data and check the shapes
for data_name in data_sets.keys():
    print("\nLoading the", data_name, "data...")
    # Load the data set
    save_file = data_name + ".npz"
    regression_data = np.load(save_file)
    X = regression_data["X"]
    y = regression_data["y"]
    print("The data set has", X.shape[0], "rows and", X.shape[1], "columns.")
