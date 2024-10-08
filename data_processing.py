"""Build the split and scaled training, validation and testing data.

Functions
---------
"""
import numpy as np
import file_methods
import xarray as xr
import importlib as imp
import pandas as pd
import pickle

imp.reload(file_methods)

__author__ = "Elizabeth A. Barnes, Senne Van Loon, and Maria Rugenstein"
__version__ = "7 October 2024"

from DIRECTORIES import DATA_DIRECTORY, MODEL_DIRECTORY


def get_members(n_train_val_test):
    n_train = n_train_val_test[0]
    n_val = n_train_val_test[1]
    n_test = n_train_val_test[2]
    all_members = np.arange(0, n_train + n_val + n_test)

    return n_train, n_val, n_test, all_members

def get_cmip_data(directory, settings, n_train_val_test, verbose=1,get_forcing=False):

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

    if settings["input_var"][-3:] == 'all':
        # MULTIPLE MODEL INPUT
        settings["data_period"] = ""
        input_var = settings["input_var"][:-3]
        label_var = settings["label_var"][:-3]
    else:
        # SINGLE MODEL INPUT
        settings["all_models"] = ["", ]
        input_var = settings["input_var"]
        label_var = settings["label_var"]

    # LOOP OVER ALL MODELS
    for im, model in enumerate(settings["all_models"]):
        # Define model years
        if "data_period" in settings.keys() and not settings["data_period"]=="":
            if settings["data_period"] == "_150":
                settings["model_yrs"] = (0,149)
            else:
                settings["model_yrs"] = (int(settings["data_period"][1:5]), int(settings["data_period"][6:10]))
        elif settings["input_var"][-3:] == 'all':
            settings["model_yrs"] = (int(model[-9:-5]), int(model[-4:]))
        elif settings["detrend"]=="1pctCO2":
            settings["model_yrs"] = (0,149)
        else:
            settings["model_yrs"] = (1870,2099)

        # Anomalies for multiple models
        if "anomaly_list" in settings.keys():
            settings["anomalies"] = settings["anomaly_list"][im]
            
        # LOAD THE DATA
        da = file_methods.get_simulations(\
            input_var+model, \
            directory, \
            all_members, \
            settings \
        )
        f_labels, labels_mean = get_labels(\
            directory, \
            all_members, \
            settings, \
            label_var=label_var+model \
        )

        # PROCESS THE INPUTS
        da, norm_dict = process_inputs(settings, da)
        norm_dict["labels_mean"] = labels_mean

        # GRAB YEARS TO TRAIN
        years = np.arange(settings["model_yrs"][0], settings["model_yrs"][1] + 1)
        iyears = np.where((years >= settings["yr_bounds"][0]) & (years <= settings["yr_bounds"][1]))[0]
        da = da[:, iyears, :, :]
        f_labels = f_labels[:, iyears]

        if im==0:
            # SPLIT THE DATA
            data_train, labels_train, data_val, labels_val, data_test, labels_test = split_data(da, f_labels, train_members, val_members, test_members)
        else:
            # SPLIT THE DATA
            data_train_i, labels_train_i, data_val_i, labels_val_i, data_test_i, labels_test_i = split_data(da, f_labels, train_members, val_members, test_members)

            # CONCATENATE WITH OTHER MODELS
            data_train = np.concatenate((data_train,data_train_i),axis=0)
            data_val = np.concatenate((data_val,data_val_i),axis=0)
            data_test = np.concatenate((data_test,data_test_i),axis=0)
            labels_train = np.concatenate((labels_train,labels_train_i),axis=0)
            labels_val = np.concatenate((labels_val,labels_val_i),axis=0)
            labels_test = np.concatenate((labels_test,labels_test_i),axis=0)

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

    # Shift data to be around zero
    if not "subtract_val" in settings.keys():
        settings["subtract_val"] = False
    if settings["subtract_val"]:
        if settings["subtract_val"] == 'save':
            print("** Saving training mean.")
            x_subtract = x_train.mean(axis=0)
            y_subtract = y_train.mean(axis=0)
            model_name = file_methods.get_model_name(settings)
            with open(MODEL_DIRECTORY+model_name+".pickle", 'wb') as f:
                pickle.dump(x_subtract, f)
                pickle.dump(y_subtract, f)
        elif settings["subtract_val"] == 'load':
            print("** Loading training mean.")
            model_name = file_methods.get_model_name(settings)
            with open(MODEL_DIRECTORY+model_name+".pickle", 'rb') as f:
                x_subtract = pickle.load(f)
                y_subtract = pickle.load(f)
        elif settings["subtract_val"][-7:] == ".pickle":
            print("** Loading training mean.")
            with open(MODEL_DIRECTORY+settings["subtract_val"], 'rb') as f:
                x_subtract = pickle.load(f)
                y_subtract = pickle.load(f)
        x_train += -x_subtract[np.newaxis,:,:,:]
        x_val   += -x_subtract[np.newaxis,:,:,:]
        x_test  += -x_subtract[np.newaxis,:,:,:]
        y_train += -y_subtract[np.newaxis,:]
        y_val   += -y_subtract[np.newaxis,:]
        y_test  += -y_subtract[np.newaxis,:]

    if get_forcing:

        F = load_forcing(directory, label_var, settings)

        return x_train, x_val, x_test, y_train, y_val, y_test, lat, lon, map_shape, member_shape, time_shape, norm_dict, member_enrollment, settings, F
    else:
        return x_train, x_val, x_test, y_train, y_val, y_test, lat, lon, map_shape, member_shape, time_shape, norm_dict, member_enrollment, settings

def load_forcing(directory, label_var, settings):

    if not settings["all_models"][0] == "":
        raise RuntimeError("Multimodel input forcing not supported.")
    
    F_filename = directory + settings["datafolder"] + 'F' + label_var[1:] + settings["data_period"] + '.nc'
    print("Loading forcing "+F_filename)

    da_F = xr.open_dataarray(F_filename)
    F = da_F.to_numpy()

    # Select only years that also have R
    years = np.arange(settings["model_yrs"][0], settings["model_yrs"][1] + 1)
    iyears = np.where((years >= settings["yr_bounds"][0]) & (years <= settings["yr_bounds"][1]))[0]

    F = F[iyears]
    years = years[iyears]

    # Anomalies
    mean_iyears = np.where(\
        (years >= settings["anomalies_years"][0]) & \
        (years <= settings["anomalies_years"][1]))[0]
    F_mean = F[mean_iyears].mean()
    F = F - F_mean

    return F

def process_inputs(settings, da):

    # Calculate anomalies
    norm_dict = {"mean": None, "std": None}
    if settings["anomalies"] is True:
        da_mean = da.sel(year=slice(settings["anomalies_years"][0], settings["anomalies_years"][1])).mean(axis=(0, 1))
        da = (da - da_mean)
        norm_dict["mean"] = da_mean.to_numpy()

    elif settings["anomalies"] == 'years':
        # Use actual years
        years = np.arange(settings["model_yrs"][0], settings["model_yrs"][1] + 1)
        iyears = np.where((years >= settings["anomalies_years"][0]) & (years <= settings["anomalies_years"][1]))[0]
        da_mean = da.sel(year=slice(iyears[0], iyears[-1])).mean(axis=(0, 1))
        da = (da - da_mean)
        norm_dict["mean"] = da_mean.to_numpy()

    elif settings["anomalies"] == 'years_member':
        # Use actual years
        years = np.arange(settings["model_yrs"][0], settings["model_yrs"][1] + 1)
        iyears = np.where((years >= settings["anomalies_years"][0]) & (years <= settings["anomalies_years"][1]))[0]
        da_mean = da.sel(year=slice(iyears[0], iyears[-1])).mean(dim="year")
        da = (da - da_mean)
        norm_dict["mean"] = da_mean.to_numpy()

    elif settings["anomalies"][-7:] == ".pickle":
        # Use saved normalization file

        norm_filename = DATA_DIRECTORY+settings["anomalies"]
        print("loading normalization from " + norm_filename)
        with open(norm_filename, 'rb') as handle:
            base_norm_dict = pickle.load(handle)

        da = (da - base_norm_dict["mean"])
        norm_dict = base_norm_dict

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


def get_labels(directory, members, settings, label_var = None):

    if label_var is None:
        radiation = file_methods.get_simulations(settings["label_var"], directory, members, settings)
    else:
        radiation = file_methods.get_simulations(label_var, directory, members, settings)

    if settings["anomalies"] is True:
        radiation_mean = radiation.sel(year=slice(settings["anomalies_years"][0], settings["anomalies_years"][1])).mean(axis=(0, 1))
        radiation = (radiation - radiation_mean)

    elif settings["anomalies"] == "years":
        # Use actual years for anomalies
        years = np.arange(settings["model_yrs"][0], settings["model_yrs"][1] + 1)
        iyears = np.where((years >= settings["anomalies_years"][0]) & (years <= settings["anomalies_years"][1]))[0]
        radiation_mean = radiation.sel(year=slice(iyears[0], iyears[-1])).mean(axis=(0, 1))
        radiation = (radiation - radiation_mean)

    elif settings["anomalies"] == "years_member":
        # Use actual years for anomalies
        years = np.arange(settings["model_yrs"][0], settings["model_yrs"][1] + 1)
        iyears = np.where((years >= settings["anomalies_years"][0]) & (years <= settings["anomalies_years"][1]))[0]
        radiation_mean = radiation.sel(year=slice(iyears[0], iyears[-1])).mean(dim="year")
        radiation = (radiation - radiation_mean)

    elif settings["anomalies"][-7:] == ".pickle":
        # Use saved normalization file
        norm_filename = DATA_DIRECTORY+settings["anomalies"]

        print("loading normalization from " + norm_filename)
        with open(norm_filename, 'rb') as handle:
            base_norm_dict = pickle.load(handle)

        radiation_mean = base_norm_dict["labels_mean"].to_numpy()
        radiation = (radiation - radiation_mean)

    else:
        radiation_mean = None

    labels = radiation.values

    return labels, radiation_mean


def get_obs_data(directory, settings, get_latlon=False):

    # Load labels
    labels_obs = xr.load_dataarray(\
        directory + settings['datafolder'] + settings['label_var'] + '.nc')
    
    # Anomalies for labels
    labels_years = labels_obs["year"].values
    labels_iyears = np.where(\
        (labels_years >= settings["anomalies_years"][0]) & \
        (labels_years <= settings["anomalies_years"][1]))[0]
    labels_mean = labels_obs[labels_iyears].mean()
    labels_obs = labels_obs - labels_mean

    # Load input data
    x_obs = xr.load_dataarray(\
        directory + settings['datafolder'] + settings['input_var'] + ".nc")
    years_obs = x_obs['year'].to_numpy()

    # Masking
    if "input_region" in settings.keys():
        if settings["input_region"][-3:] == '.nc':
            mask = xr.load_dataarray(DATA_DIRECTORY + 'shapefiles/' + settings["input_region"]).to_numpy()
            x_obs = x_obs * mask
            x_obs = x_obs.fillna(0)
        else:
            RuntimeError("Masking not supported.")
    lat = x_obs['lat'].to_numpy()
    lon = x_obs['lon'].to_numpy()
    x_obs = x_obs.fillna(0)
    x_obs = x_obs.to_numpy()

    # Anomalies for input
    iyears = np.where(\
        (years_obs >= settings["anomalies_years"][0]) & \
        (years_obs <= settings["anomalies_years"][1]))[0]
    x_mean = np.mean(x_obs[iyears, :, :], axis=0)
    x_obs = x_obs - x_mean

    if not "subtract_val" in settings.keys():
        settings["subtract_val"] = False
    if settings["subtract_val"]:
        if settings["subtract_val"][-7:] == ".pickle":
            print("** Loading training mean. "+settings["subtract_val"][:-7])
            with open(MODEL_DIRECTORY+settings["subtract_val"], 'rb') as f:
                x_subtract = pickle.load(f)
            x_obs += -x_subtract[np.newaxis,:,:,0]
        else:
            RuntimeError("Value subtraction not supported.")

    if get_latlon:
        return labels_obs, x_obs, years_obs, lat, lon
    else:
        return labels_obs, x_obs, years_obs
