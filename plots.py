"""Metrics for generic plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.feature as cfeature
from shapely.geometry.polygon import LinearRing
import cartopy.crs as ccrs

import cartopy as ct
import warnings
import sklearn

warnings.filterwarnings("ignore")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150

data_crs = ccrs.PlateCarree()
proj_Global = ccrs.EqualEarth(central_longitude=200)

def savefig(filename, dpi=300):
    for fig_format in (".png", ".pdf"):
        plt.savefig(filename + fig_format,
                    bbox_inches="tight",
                    dpi=dpi)


def round_down_ten(x):
    y = np.floor(x / 10.) * 10.
    return y


def plot_metrics(history, metric):
    imin = np.argmin(history.history['val_loss'])

    plt.plot(history.history[metric], label='training')
    plt.plot(history.history['val_' + metric], label='validation')
    plt.title(metric)
    plt.axvline(x=imin, linewidth=.5, color='gray', alpha=.5)
    plt.legend()


def plot_metrics_panels(history, predictions):
    # BASELINES TO BEAT
    baseline_mae = np.mean(np.abs(predictions["labels_test"] - 0.0))
    baseline_mse = np.mean((predictions["labels_test"] - 0.0)**2)

    # DO THE PLOTTING
    plt.subplots(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    plot_metrics(history, 'loss')
    plt.ylim(0, .5)
    plt.axhline(y=baseline_mse, color="gray", linestyle="--", label="baseline")

    plt.subplot(1, 4, 2)
    plot_metrics(history, "mae")
    plt.ylim(0, .5)
    plt.axhline(y=baseline_mae, color="gray", linestyle="--", label="baseline")

    plt.legend()


def plot_pred_vs_truth(predictions, settings, ms=5, label=''):

    for ivar, var in enumerate(settings["label_vars"]):
        mse = sklearn.metrics.mean_squared_error(predictions["labels_val"][:, 0], predictions["pred_val"][:, 0])
        if np.isnan(mse):
            mse = 0.
        plt.plot(predictions["labels_val"][:, ivar], predictions["pred_val"][:, ivar], '.', alpha=.5,
                 label=label + ' (MSE=' + str(mse.round(3)) + ')', markersize=ms)

    plt.plot((-10, 10), (-10, 10), '--k', alpha=.5)
    plt.ylim(-8, 3)
    plt.xlim(-8, 3)
    plt.ylabel('predicted')
    plt.xlabel('truth')
    plt.legend(frameon=False, fontsize=8)
    plt.gca().set_aspect('equal')


def setup_figure(nCols=1,nRows=1,size=(15,15),mask=True):

    map_proj = proj_Global

    land_feature = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        facecolor='gray',
        edgecolor='k',
        linewidth=.25,
    )

    fig = plt.figure(figsize=size)
    if nCols==1 and nRows==1:
        ax = fig.add_subplot(1, 1, 1, projection=map_proj)

        ax.coastlines('50m', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.gridlines(
            draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
        )

        if mask:
            ax.add_feature(land_feature)
    elif nCols > 1 and nRows > 1:
        ax = np.empty((nCols,nRows),dtype=object)
        iF = 1
        for jj in range(nRows):
            for ii in range(nCols):
                ax[ii,jj] = fig.add_subplot(nRows, nCols, iF, projection=map_proj)

                ax[ii,jj].coastlines('50m', linewidth=0.8)
                ax[ii,jj].tick_params(axis='both', which='major', labelsize=10)
                ax[ii,jj].gridlines(
                    draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
                )
                if mask:
                    ax[ii,jj].add_feature(land_feature)
                iF += 1
    elif nCols > 1:
        ax = np.empty((nCols),dtype=object)
        for ii in range(nCols):
            ax[ii] = fig.add_subplot(1, nCols, ii+1, projection=map_proj)

            ax[ii].coastlines('50m', linewidth=0.8)
            ax[ii].tick_params(axis='both', which='major', labelsize=10)
            ax[ii].gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            if mask:
                ax[ii].add_feature(land_feature)
    elif nRows > 1:
        ax = np.empty((nRows),dtype=object)
        for ii in range(nRows):
            ax[ii] = fig.add_subplot(nRows, 1, ii+1, projection=map_proj)

            ax[ii].coastlines('50m', linewidth=0.8)
            ax[ii].tick_params(axis='both', which='major', labelsize=10)
            ax[ii].gridlines(
                draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
            )
            if mask:
                ax[ii].add_feature(land_feature)

            
    return fig, ax


def add_mask(ax,region,lon=None,lat=None):
    
    if region == "land":
        land_feature = cfeature.NaturalEarthFeature(
            category='physical',
            name='land',
            scale='50m',
            facecolor='gray',
            edgecolor='k',
            linewidth=.25,
        )
        ax.add_feature(land_feature)
    elif region[:4] == "reg_":

        regions_range_dict = regions.get_region_dict(region)

        min_lon, max_lon = regions_range_dict["lon_range"]
        min_lat, max_lat = regions_range_dict["lat_range"]
        region = [min_lon, min_lat, max_lon, max_lat]

        add_square(ax,region,data_crs,facecolor='gray')

    elif region[-3:] == ".nc":

        mask = xr.load_dataarray(SHAPE_DIRECTORY + region).to_numpy()

        mask[mask>0.5] = np.nan
        mask_cyc, lons_cyc = add_cyclic_point(mask, coord=lon)
        ax.pcolormesh(lons_cyc,lat,mask_cyc,cmap="gray")

    return ax


def add_square(ax,region,crs,**kwargs):
    # Region = lon1, lat1; lon2, lat2

    lons = [region[0],region[2],region[2],region[0]]
    lats = [region[1],region[1],region[3],region[3]]
    ring = LinearRing(list(zip(lons,lats)))

    ax.add_geometries([ring],crs,**kwargs)

    return ax


def add_loc_square(ax,settings,**kwargs):
    if "mask_region" in settings.keys():
        maskout=settings["mask_region"]
    else:
        return ax

    if maskout == "indonesia":
        min_lon, max_lon = 75., 130.
        min_lat, max_lat = -15., 10.
    elif maskout == "westpacific":
        min_lon, max_lon = 170., 230.
        min_lat, max_lat = -30., -5.
    elif maskout == "eastpacific":
        min_lon, max_lon = 210., 280.
        min_lat, max_lat = -10., 10.
    elif maskout == "caribbean":
        min_lon, max_lon = 250., 320.
        min_lat, max_lat = 5., 25.
    elif maskout == "brazil":
        min_lon, max_lon = 310., 350.
        min_lat, max_lat = -25., -5.
    elif maskout == "namibia":
        min_lon, max_lon = 0., 20.
        min_lat, max_lat = -45., -10.
    else:
        raise NotImplementedError("no such mask type.")
    
    if min_lon > 180 and max_lon > 180:
        min_lon = min_lon - 360
        max_lon = max_lon - 360
    
    add_square(ax,[min_lon,min_lat,max_lon,max_lat],data_crs,**kwargs)

    return ax

def round_to_n(x,n):
    if x == 0:
        return x
    else:
        return np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
    
def num_lab(x,n):
    return str(round_to_n(x,n))