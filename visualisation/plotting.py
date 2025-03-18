import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import glasbey


# Potentially remove this one again?
def plot_each_depth_level(df, color_label="water", save_as="output/grid_cells.png"):
    # Show grid cells for every depth layer
    depths = df["LEV_M"].unique()
    lamin = df["LATITUDE"].min() - 5
    lamax = df["LATITUDE"].max() + 5
    lomin = df["LONGITUDE"].min() - 5
    lomax = df["LONGITUDE"].max() + 5

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(2 * 6, 10), sharex=True, sharey=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    for i in range(len(depths)):
        d = depths[i]
        sel = df[df["LEV_M"] == d]

        axs[i].coastlines()
        axs[i].gridlines(draw_labels=True)
        axs[i].set_extent([lomin, lomax, lamin, lamax], crs=ccrs.PlateCarree())
        sc = axs[i].scatter(sel["LONGITUDE"], sel["LATITUDE"], s=0.5, c=sel[color_label])
        axs[i].set_title(f"{d}m")

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show(block=True)


def plot_boxplots(df, parameters, save_as):
    pass


def plot_histograms(df, save_as=None):
    pass


def plot_embedding(embedding, color_label=None, alpha=0.08, size=2, save_as=None, figsize=(6, 6), fontsize=8,
                   ticklabelsize=8, dpi=1000):
    """ Plot a 2d or 3d embedding. (It can be a pandas.DataFrame or a numpy.array.) """
    # determine x, y, z, data and colour
    if isinstance(embedding, pd.DataFrame):
        x = embedding["e0"]
        y = embedding["e1"]
        z = embedding["e2"] if "e2" in embedding.columns else None
        c = embedding[color_label] if color_label is not None else None
    else:
        x = embedding[:, 0]
        y = embedding[:, 1]
        z = embedding[:, 2] if embedding.shape[1] == 3 else None
        c = color_label

        # plot
    fig = plt.figure(figsize=figsize)

    # 3d
    if z is not None:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel("Z-axis")
        if c is None:
            ax.scatter(x, y, z, alpha=alpha, s=size, marker=".")
        else:
            ax.scatter(x, y, z, alpha=alpha, c=c, s=size, marker=".")

        plt.subplots_adjust(left=-0.05, right=0.9, top=1.1, bottom=0)
    # 2d
    else:
        if c is None:
            plt.scatter(x, y, alpha=alpha, s=size, marker=".")
        else:
            plt.scatter(x, y, alpha=alpha, c=c, s=size, marker=".")

    plt.xlabel("X-axis", fontsize=fontsize)
    plt.ylabel("Y-axis", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=ticklabelsize)  # , pad=tick_padding)
    ax.tick_params(axis='y', labelsize=ticklabelsize)  # , pad=tick_padding)
    ax.tick_params(axis='z', labelsize=ticklabelsize)  # , pad=tick_padding)
    plt.tight_layout()
    plt.subplots_adjust(left=-0.01, right=0.92, top=1.1, bottom=0)
    if save_as:
        plt.savefig(save_as, dpi=dpi)
    plt.show()


def color_code_labels(df, color_noise_black=False, drop_noise=False, column_name="label"):
    """ Add a color for each label in the clustering using the Glasbey library. """
    temp = df.copy()

    # define colors
    unique_labels = np.sort(np.unique(temp[column_name]))
    colors = glasbey.create_palette(palette_size=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    temp["color"] = temp[column_name].map(lambda x: color_map[x])

    # how to deal with -1 labels (which is noise in DBSCAN)
    if color_noise_black:
        temp.loc[temp[column_name] == -1, "color"] = "#000000"
    if drop_noise:
        temp = temp[temp[column_name] != -1]

    return temp

