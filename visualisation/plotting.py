import matplotlib.pyplot as plt
import cartopy.crs as ccrs


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
