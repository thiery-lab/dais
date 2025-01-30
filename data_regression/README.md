## Scripts for the regression examples

* `preprocess_data.py`: Downloads the data and preprocesses it. Results are
  stored in the numpy files `spam.npz`, `krkp.npz`, `ionosphere.npz` and
  `mushroom.npz`.
* `regression_logdensity.py`: Defines the posterior log-density function for the
  regression model. It is loaded by `regression_HMC.py`.
* `regression_HMC.py`: Runs BlackJAX' HMC sampler on the regression model. It
  reads in the .npz files produced by `preprocess_data.py` and saves the HMC
  output as .npz files in `./mcmc_output/`.

**Remark**: The HMC outputs are stored in the `./mcmc_output/` folder. They
  can be recreated using the following shell script:
```
python preprocess_data.py
python regression_HMC.py
```
