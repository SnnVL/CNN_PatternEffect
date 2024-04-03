import gc
import os
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xarray as xr

import data_processing
import experiment_settings
import file_methods
import network
import plots
import cartopy as ct
import sklearn
from sklearn import metrics

plt.style.use("default")
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150
savefig_dpi = 300
# np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"tensorflow version = {tf.__version__}")

# ---------------------------------------------------------
# ---------------------------------------------------------

EXP_NAME_LIST = ("MPI_GF",)
OVERWRITE_MODEL = True

from DIRECTORIES import MODEL_DIRECTORY, PREDICTIONS_DIRECTORY, DATA_DIRECTORY, FIGURES_DIRECTORY

# ---------------------------------------------------------
greens_weights = xr.load_dataarray(DATA_DIRECTORY+"rad_gf.nc").values[::-1, :]
greens_weights = np.ndarray.flatten(greens_weights)
# ---------------------------------------------------------

for EXP_NAME in EXP_NAME_LIST:
    print("------" + EXP_NAME + "------")
    settings = experiment_settings.get_settings(EXP_NAME)
    # display(settings)

    for rng_seed in settings["rng_seed_list"]:
        settings["rng_seed"] = rng_seed
        tf.random.set_seed(settings["rng_seed"])
        random.seed(settings["rng_seed"])
        np.random.seed(settings["rng_seed"])

        # GET MODEL NAME AND CHECK IF IT EXISTS
        model_name = file_methods.get_model_name(settings)
        if os.path.exists(MODEL_DIRECTORY + model_name + "_model") and OVERWRITE_MODEL is False:
            print("\n" + model_name + "exists. Skipping...")
            print("================================\n")
            continue

            # GET THE DATA
        (
            x_train,
            x_val,
            x_test,
            labels_train,
            labels_val,
            labels_test,
            lat,
            lon,
            map_shape,
            member_shape,
            time_shape,
            norm_dict,
            member_enrollment,
            settings,
        ) = data_processing.get_cmip_data(
            DATA_DIRECTORY,
            settings,
            n_train_val_test=settings["n_train_val_test"],
        )

        # ----------------------------------------
        tf.keras.backend.clear_session()
        # define early stopping callback (cannot be done elsewhere)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=settings["patience"], verbose=1, mode="auto", restore_best_weights=True
        )

        model = network.compile_model(x_train, labels_train, settings)
        model.summary()
        model.layers[3].set_weights([greens_weights[:, np.newaxis], np.zeros((1,))])
        model.layers[4].set_weights([np.ones((1, 1)), np.zeros((1,))])

        # weights_in = np.copy(model.layers[3].get_weights()[0])

        history = model.fit(
            x_train,
            labels_train,
            epochs=settings["n_epochs"],
            batch_size=settings["batch_size"],
            shuffle=True,
            validation_data=[x_val, labels_val],
            callbacks=[
                early_stopping,
            ],
            verbose=2,
        )

        # make predictions dictionary
        predictions = {
            "labels_train": labels_train,
            "pred_train": model.predict(x_train),
            "labels_val": labels_val,
            "pred_val": model.predict(x_val),
            "labels_test": labels_test,
            "pred_test": model.predict(x_test),
        }

        # clean-up from model.predict
        _ = gc.collect()

        # ----------------------------------------
        # save the tensorflow model and predictions
        file_methods.save_tf_model(model, model_name, MODEL_DIRECTORY, settings)
        file_methods.save_predictions(predictions, PREDICTIONS_DIRECTORY + model_name + "_predictions.pickle")

        plots.plot_pred_vs_truth(predictions, settings)
        plt.savefig(FIGURES_DIRECTORY + model_name + "_pred_vs_truth" + ".png", dpi=savefig_dpi, bbox_inches="tight")
        plt.close()

        # #----------------------------------------
        # save predictions over all members to use as labels for other networks
        (x_all, __, __, labels_all, __, __, __, __, __, __, __, __, __, settings) = data_processing.get_cmip_data(
            DATA_DIRECTORY,
            settings,
            n_train_val_test=(settings["n_members"], 0, 0),
        )

        pred_all = model.predict(x_all)
        pred_all = pred_all.reshape(settings["n_members"], time_shape, 1)
        labels_all = labels_all.reshape(settings["n_members"], time_shape, 1)

        predictions_all = {"pred_all": pred_all, "labels_all": labels_all}
        file_methods.save_predictions(predictions_all, PREDICTIONS_DIRECTORY + model_name + "_all_predictions.pickle")

        # clean-up from model.predict
        _ = gc.collect()
