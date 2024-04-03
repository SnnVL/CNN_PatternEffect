"""Build the split and scaled training, validation and testing data.

Functions
---------
"""
import numpy as np
import file_methods
import xarray as xr
import importlib as imp
import pandas as pd

imp.reload(file_methods)

__author__ = "Elizabeth A. Barnes and Maria Rugenstein"
__version__ = "3 April 2024"

from DIRECTORIES import DATA_DIRECTORY


def get_members(n_train_val_test):
    n_train = n_train_val_test[0]
    n_val = n_train_val_test[1]
    n_test = n_train_val_test[2]
    all_members = np.arange(0, n_train + n_val + n_test)

    return n_train, n_val, n_test, all_members

def maskout_land_ocean(da, maskout="land"):
    # if no land mask or ocean masks exists, run make_land_ocean_mask()
    if maskout == "land":
        mask = xr.load_dataarray(DATA_DIRECTORY + "MPI-ESM_ocean_mask.nc").to_numpy()
    elif maskout == "ocean":
        mask = xr.load_dataarray(DATA_DIRECTORY + "MPI-ESM_land_mask.nc").to_numpy()
    else:
        raise NotImplementedError("no such mask type.")

    return da * mask

def get_cmip_data(directory, settings, n_train_val_test, verbose=1):

    # DEFINE TRAINING / VALIDATION / TESTING MEMBERS
    assert np.sum(settings["n_train_val_test"]) == settings["n_members"]
    n_train, n_val, n_test, all_members = get_members(n_train_val_test)

    # Select members for training/validation/testing
    rng_cmip = np.random.default_rng(settings["rng_seed"])
    train_members = np.sort(rng_cmip.choice(all_members, size=n_train, replace=False))
    val_members = np.sort(rng_cmip.choice(np.setdiff1d(all_members, train_members), size=n_val, replace=False))
    test_members = np.sort(rng_cmip.choice(np.setdiff1d(all_members, np.append(train_members[:], val_members)), size=n_test, replace=False))
    member_enrollment = (train_members, val_members, test_members)
    if verbose == 1:
        print(member_enrollment)

    # save the meta data
    settings['train_members'] = train_members.tolist()
    settings['val_members'] = val_members.tolist()
    settings['test_members'] = test_members.tolist()

    # LOAD THE DATA
    da = file_methods.get_simulations(\
        settings["input_var"], \
        directory, \
        all_members, \
        settings \
    )
    f_labels, labels_mean = get_labels(\
        settings["label_var"], \
        directory, \
        all_members, \
        settings, \
    )

    # PROCESS THE INPUTS
    da, norm_dict = process_inputs(settings, da)
    norm_dict["labels_mean"] = labels_mean

    # SPLIT THE DATA 
    data_train, labels_train, data_val, labels_val, data_test, labels_test = split_data(da, f_labels, train_members, val_members, test_members)

    # RESHAPE TO SAMPLES
    x_train = data_train.reshape((data_train.shape[0] * data_train.shape[1], data_train.shape[2], data_train.shape[3], 1))
    x_val = data_val.reshape((data_val.shape[0] * data_val.shape[1], data_val.shape[2], data_val.shape[3], 1))
    x_test = data_test.reshape((data_test.shape[0] * data_test.shape[1], data_test.shape[2], data_test.shape[3], 1))

    y_train = labels_train.reshape((data_train.shape[0] * data_train.shape[1], labels_train.shape[-1]))
    y_val = labels_val.reshape((data_val.shape[0] * data_val.shape[1], labels_val.shape[-1]))
    y_test = labels_test.reshape((data_test.shape[0] * data_test.shape[1], labels_test.shape[-1]))

    if len(y_train.shape) == 1:
        y_train = y_train[:, np.newaxis]
        y_val = y_val[:, np.newaxis]
        y_test = y_test[:, np.newaxis]

    if verbose == 1:
        print('---')
        print(f"training.shape = {x_train.shape, y_train.shape}")
        print(f"validation.shape = {x_val.shape, y_val.shape}")
        print(f"testing.shape = {x_test.shape, y_test.shape}")

    # DEFINE FINAL VALUES
    lat, lon = da["lat"].values, da["lon"].values
    map_shape = np.shape(data_train)[2:]
    member_shape = data_train.shape[0]
    time_shape = data_train.shape[1]

    return x_train, x_val, x_test, y_train, y_val, y_test, lat, lon, map_shape, member_shape, time_shape, norm_dict, member_enrollment, settings

def process_inputs(settings, da):

    # Calculate anomalies
    norm_dict = {"mean": None, "std": None}
    if settings["anomalies"]:
        da_mean = da.sel(year=slice(settings["anomalies_years"][0], settings["anomalies_years"][1])).mean(axis=(0, 1))
        da = (da - da_mean)
        norm_dict["mean"] = da_mean.to_numpy()

    # Maskout land
    mask = xr.load_dataarray(DATA_DIRECTORY + settings["input_region"]).to_numpy()
    da = da * mask
    da = da.fillna(0)

    return da, norm_dict


def split_data(da, f_labels, train_members, val_members, test_members):

    # TRAINING
    data_train = da[train_members, :, :, :].values
    labels_train = f_labels[train_members, :][:, :, np.newaxis]
    print(np.shape(data_train), np.shape(labels_train))

    # VALIDATION
    data_val = da[val_members, :, :, :].values
    labels_val = f_labels[val_members, :][:, :, np.newaxis]
    print(np.shape(data_val), np.shape(labels_val))

    # TESTING
    data_test = da[test_members, :, :, :].values
    labels_test = f_labels[test_members, :][:, :, np.newaxis]
    print(np.shape(data_test), np.shape(labels_test))

    return data_train, labels_train, data_val, labels_val, data_test, labels_test


def get_labels(label_var, directory, members, settings):

    radiation = file_methods.get_simulations(label_var, directory, members, settings)

    if settings["anomalies"]:
        radiation_mean = radiation.sel(year=slice(settings["anomalies_years"][0], settings["anomalies_years"][1])).mean(axis=(0, 1))
        radiation = (radiation - radiation_mean)
    else:
        radiation_mean = None

    labels = radiation.values

    return labels, radiation_mean