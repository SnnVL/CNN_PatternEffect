"""Functions for working with generic files.

Functions
---------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)
save_tf_model(model, model_name, directory, settings)
get_cmip_filenames(settings, verbose=0)
"""

import xarray as xr
import json
import pickle
import tensorflow as tf

__author__ = "Elizabeth A. Barnes and Noah Diffenbaugh"
__version__ = "07 February 2023"


def get_model_name(settings, suffix=""):
    model_name = (settings["exp_name"] + suffix + '_rngseed' + str(settings["rng_seed"]))
    return model_name


def get_netcdf_da(filename, members, settings):
    print(filename)
    try:
        return xr.open_dataarray(filename)[members, :, :, :]
    except:
        return xr.open_dataarray(filename)[members, :]


def save_predictions(predictions, filename):
    with open(filename, 'wb') as f:
        pickle.dump(predictions, f)


def load_predictions(filename):
    with open(filename, 'rb') as f:
        predictions = pickle.load(f)
    return predictions


def load_tf_model(model_name, directory):
    # loading a tf model
    model = tf.keras.models.load_model(
        directory + model_name + "_model",
        compile=False,
    )
    return model


def save_tf_model(model, model_name, directory, settings):
    # save the tf model
    tf.keras.models.save_model(model, directory + model_name + "_model", overwrite=True)

    # save the meta data
    with open(directory + model_name + '_metadata.json', 'w') as json_file:
        json_file.write(json.dumps(settings))


def get_simulations(var, directory, members, settings):

    if settings["detrend"] is True:
        v = get_netcdf_da(directory + settings["datafolder"] + var \
                          + settings["data_period"] + "_detrend.nc", members, settings)
    elif settings["detrend"] == "1pctCO2":
        v = get_netcdf_da(directory + settings["datafolder"] + var \
                          + "_150.nc", members, settings)
    else:
        v = get_netcdf_da(directory + settings["datafolder"] + var \
                          + settings["data_period"] + ".nc", members, settings)
    return v
