"""Experimental settings


Functions
---------
get_settings(experiment_name)
"""
import numpy as np

__author__ = "Maria Rugenstein, Elizabeth A. Barnes, and Senne Van Loon"
__date__ = "7 October 2024"


def get_settings(experiment_name):
    experiments = {

        # Green's function
        "MPI_GF": {  
            "input_region": "mask_MPI.nc",
            "datafolder": "MPI_hist_rcp85/",
            "detrend": True,
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": "tas", 
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.0, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": [3, ],
            "n_epochs": 0,
        },

        # MPI CNN used in Rugenstein et al. (2024); Hyperparameter tuning
        "MPI_IV_hpt0": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32, 32],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [16, 8],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt1": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32],
            "kernel_act": ["relu", "relu"],
            "hiddens": [16, 8],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.00005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt2": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32],
            "kernel_act": ["relu", "relu"],
            "hiddens": [16, 8],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.0000025,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt3": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32],
            "kernel_act": ["relu", "relu"],
            "hiddens": [16, 8],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt4": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [64, 64, ],
            "kernel_act": ["relu", "relu", ],
            "hiddens": [16, 8],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt5": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32, ],
            "kernel_act": ["relu", "relu", ],
            "hiddens": [8, 8, 8],
            "act_fun": ["elu", "elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt6": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32, ],
            "kernel_act": ["relu", "relu", ],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt7": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32, ],
            "kernel_act": ["relu", "relu", ],
            "hiddens": [16, 8],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            # "rng_seed_list": np.arange(5).tolist(),
            "rng_seed_list": [3,],
            "n_epochs": 25_000,
        },
        "MPI_IV_hpt8": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32, ],
            "kernel_act": ["relu", "relu", ],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MPI_IV_linear": {
            "input_region": "mask_MPI.nc",
            "detrend": True,
            "datafolder": "hist_rcp85/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation",
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.25, ],
            "dropout_rate": [0.25, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            # "rng_seed_list": np.arange(5).tolist(),
            "rng_seed_list": [3,],
            "n_epochs": 25_000,
        },

        # Single model CNNs + Linear networks used in Rugenstein et al. (2024)
        "SingleModel_MPI": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": "tas_MPI-ESM", 
            "label_var": "R_MPI-ESM",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "ridge_param": [0.0, 0.0],
            "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_CanESM5": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1850-2100",
            "input_var": "tas_CanESM5", 
            "label_var": "R_CanESM5",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "ridge_param": [0.0, 0.0],
            "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_IPSL": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1850-2059",
            "input_var": "tas_IPSL-CM6A-LR", 
            "label_var": "R_IPSL-CM6A-LR",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "ridge_param": [0.0, 0.0],
            "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_MIROC6": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1850-2039",
            "input_var": "tas_MIROC6", 
            "label_var": "R_MIROC6",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "ridge_param": [0.0, 0.0],
            "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_MPI_linear": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1870-2099",
            "input_var": "tas_MPI-ESM", 
            "label_var": "R_MPI-ESM",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.25, ],
            "dropout_rate": [0.25, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_CanESM5_linear": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1850-2100",
            "input_var": "tas_CanESM5", 
            "label_var": "R_CanESM5",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.25, ],
            "dropout_rate": [0.25, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_IPSL_linear": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1850-2059",
            "input_var": "tas_IPSL-CM6A-LR", 
            "label_var": "R_IPSL-CM6A-LR",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.25, ],
            "dropout_rate": [0.25, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "SingleModel_MIROC6_linear": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": True,
            "anomalies_years": (0, 30),

            "data_period":"_1850-2039",
            "input_var": "tas_MIROC6", 
            "label_var": "R_MIROC6",
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.25, ],
            "dropout_rate": [0.25, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },

        # MultiModel CNN used in Rugenstein et al. (2024)
        "MultiModelELU": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "kernel_act": ["relu", "relu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "elu"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModelELU_linear": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "ffn",
            "hiddens": [1, ],
            "act_fun": ["linear", ],
            "ridge_param": [0.25, ],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },


        # MultiModel CNN hyperparameter tuning. MultiModel_hpt0_s3 used in Van Loon et al. (2024)
        "MultiModel_hpt0": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "kernel_act": ["elu", "elu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModel_hpt1": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 5,
            "kernels": [32, 32],
            "kernel_act": ["elu", "elu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModel_hpt2": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [64, 64],
            "kernel_act": ["elu", "elu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModel_hpt3": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32, 32],
            "ridge_param": [0.0, 0.0, 0.0],
            "kernel_act": ["elu", "elu", "elu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModel_hpt4": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [16, 16],
            "kernel_act": ["elu", "elu"],
            "hiddens": [32, 16],
            "act_fun": ["elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModel_hpt5": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "kernel_act": ["elu", "elu"],
            "hiddens": [16, 16],
            "act_fun": ["elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },
        "MultiModel_hpt6": {
            "input_region": "mask_all.nc",
            "detrend": True,
            "datafolder": "grid_128_64/",
            "anomalies": '',
            "anomaly_list": ["norm_CanESM5.pickle", "norm_IPSL-CM6A-LR.pickle", "norm_MPI-ESM.pickle", "norm_MIROC6.pickle"],
            "subtract_val": 'save',

            "input_var": "tas_all", 
            "label_var": "R_all",
            "all_models": ["CanESM5_1850-2100", "IPSL-CM6A-LR_1850-2059", "MPI-ESM_1870-2099", "MIROC6_1850-2039"],
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "network_type": "cnn",
            "kernel_size": 3,
            "kernels": [32, 32],
            "kernel_act": ["elu", "elu"],
            "hiddens": [8, 8, 16],
            "act_fun": ["elu", "elu", "linear"],
            "learning_rate": 0.000005,
            "batch_size": 32,
            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
            "n_epochs": 25_000,
        },


        # DATA for OOS prediction
        "DATA_MPI_hist_rcp85_origGrid": {
            "input_region": "mask_MPI.nc",
            "datafolder": "hist_rcp85/",
            "anomalies": "norm_1870-1900.pickle",

            "data_period":"_1870-2099",
            "input_var": 'tas',
            "label_var": "radiation", 
            "label_type": "None",
            "n_train_val_test": (80, 10, 10),
            "yr_bounds": (1870, 2099),

            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
        },
        "DATA_MPI_hist_rcp85": {
            "input_region": "mask_all.nc",
            "detrend": False,
            "datafolder": "grid_128_64/",
            "anomalies": 'years_member',
            "anomalies_years": (2001, 2020),

            "data_period":"_1870-2099",
            "input_var": "tas_MPI-ESM", 
            "label_var": "R_MPI-ESM", 
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
        },
        "DATA_CanESM5_hist_ssp245": {
            "input_region": "mask_all.nc",
            "detrend": False,
            "datafolder": "grid_128_64/",
            "anomalies": 'years_member',
            "anomalies_years": (2001, 2020),

            "data_period":"_1850-2100",
            "input_var": "tas_CanESM5", 
            "label_var": "R_CanESM5", 
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
        },
        "DATA_IPSL_hist_ssp245": {
            "input_region": "mask_all.nc",
            "detrend": False,
            "datafolder": "grid_128_64/",
            "anomalies": 'years_member',
            "anomalies_years": (2001, 2020),

            "data_period":"_1850-2059",
            "input_var": "tas_IPSL-CM6A-LR", 
            "label_var": "R_IPSL-CM6A-LR", 
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
        },
        "DATA_MIROC6_hist_ssp245": {
            "input_region": "mask_all.nc",
            "detrend": False,
            "datafolder": "grid_128_64/",
            "anomalies": 'years_member',
            "anomalies_years": (2001, 2020),

            "data_period":"_1850-2039",
            "input_var": "tas_MIROC6", 
            "label_var": "R_MIROC6", 
            "n_train_val_test": (19,3,3),
            "yr_bounds": (1870, 2039),

            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
        },
        "DATA_1pctC02": {
            "input_region": "mask_all.nc",
            "detrend": False,
            "datafolder": "grid_128_64/",
            "anomalies": 'years_member',
            "anomalies_years": (71, 90),

            "data_period":"_150",
            "input_var": "tas_1pctC02", 
            "label_var": "R_1pctC02", 
            "n_train_val_test": (19,3,3),
            "yr_bounds": (0, 149),

            "rng_seed": None,
            "rng_seed_list": np.arange(5).tolist(),
        },

        # OBSERVATIONS
        "obs_ERA5_deepC+CERES": {
            "datafolder": "obs/",
            "input_region": "mask_all.nc",
            "anomalies_years": (2001,2020),

            "input_var": "tas_ERA5_1940_2023", 
            "label_var": "deepC+CERES_1985_2023",
        },
        "obs_ERA5_SST_deepC+CERES": {
            "datafolder": "obs/",
            "input_region": "mask_all.nc",
            "anomalies_years": (2001,2020),

            "input_var": "sst_ERA5_1940_2023", 
            "label_var": "deepC+CERES_1985_2023",
        },
        "obs_COBE2_deepC+CERES": {
            "datafolder": "obs/",
            "input_region": "mask_all.nc",
            "anomalies_years": (2001,2020),

            "input_var": "sst_COBE2_1850-2023", 
            "label_var": "deepC+CERES_1985_2023",
        },
        "obs_PCMDI_deepC+CERES": {
            "datafolder": "obs/",
            "input_region": "mask_all.nc",
            "anomalies_years": (2001,2020),

            "input_var": "PCMDI-AMIP-1-1-9_1870-2022", 
            "label_var": "deepC+CERES_1985_2023",
        },
        "obs_HadISST_deepC+CERES": {
            "datafolder": "obs/",
            "input_region": "mask_all.nc",
            "anomalies_years": (2001,2020),

            "input_var": "HadISST-1.1_1870-2022", 
            "label_var": "deepC+CERES_1985_2023",
        },

    }

    exp_dict = experiments[experiment_name]
    exp_dict['exp_name'] = experiment_name
    if 'n_train_val_test' in exp_dict.keys():
        exp_dict['n_members'] = int(np.sum(exp_dict["n_train_val_test"]))

    return exp_dict
