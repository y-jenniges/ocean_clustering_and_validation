import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import plotly.graph_objects as go
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


def plot_geo(df, color_label="color", save_as=None, figsize=(6, 6),
             adjust_left=0, adjust_right=0.92, adjust_top=1.1, adjust_bottom=-0.05, pointsize=0.5, dpi=600,
             xlabelpad=20, ylabelpad=0, zlabelpad=0):
    # Define basemap
    mymap = Basemap(llcrnrlon=df["LONGITUDE"].min(), llcrnrlat=df["LATITUDE"].min(),
                    urcrnrlon=df["LONGITUDE"].max(), urcrnrlat=df["LATITUDE"].max(), fix_aspect=False)

    # Geographical plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df["LONGITUDE"], df["LATITUDE"], df["LEV_M"], c=df[color_label], s=pointsize, alpha=1, zorder=4)
    ax.add_collection3d(mymap.drawcoastlines(linewidth=0.5))
    ax.set_box_aspect((np.ptp(df["LONGITUDE"]), np.ptp(df["LATITUDE"]), np.ptp(df["LEV_M"]) / 50))

    # Add axis labels
    ax.set_xlabel('Longitude', labelpad=xlabelpad)
    ax.set_ylabel('Latitude', labelpad=ylabelpad)
    ax.set_zlabel('Depth [m]', labelpad=zlabelpad)

    # Invert the Z-axis for depth representation
    plt.gca().invert_zaxis()

    # Define coarse ticks for longitude and latitude
    lon_ticks = np.linspace(df["LONGITUDE"].min(), df["LONGITUDE"].max(), num=5)  # Adjust num for desired spacing
    lat_ticks = np.linspace(df["LATITUDE"].min(), df["LATITUDE"].max(), num=5)

    # Set ticks and labels
    ax.set_xticks(lon_ticks)  # Longitude ticks
    ax.set_xticklabels([f"{tick:.1f}°" for tick in lon_ticks], rotation=45, ha="right")  # Format as degrees

    ax.set_yticks(lat_ticks)  # Latitude ticks
    ax.set_yticklabels([f"{tick:.1f}°" for tick in lat_ticks], rotation=0, ha="left")  # Format as degrees

    # Increase padding between ticks and their labels
    ax.tick_params(axis='x', pad=-5)  # Horizontal ticks
    ax.tick_params(axis='y', pad=-5)  # Vertical ticks
    ax.tick_params(axis='z', pad=10)  # Depth ticks

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=adjust_left, right=adjust_right, top=adjust_top, bottom=adjust_bottom)

    # Save figure
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


def plot_interactive_3d_labels_geo(df, column="label", color_label="color"):
    """ Interactive 3d geographic scatter plot. """
    df_display = df.copy()

    scatter_size = 2
    margin = 5

    longitude_min = df["LONGITUDE"].min()
    longitude_max = df["LONGITUDE"].max()
    latitude_min = df["LATITUDE"].min()
    latitude_max = df["LATITUDE"].max()
    depth_min = (-1 * df["LEV_M"].min())
    depth_max = (-1 * df["LEV_M"].max())

    # Define figure
    figure_geo = go.Figure(data=go.Scatter3d(x=df_display.LONGITUDE, y=df_display.LATITUDE, z=df_display.LEV_M * -1,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=df[color_label], opacity=1),
                                             hovertemplate='Longitude: %{x}<br>' +
                                                           'Latitude: %{y}<br>' +
                                                           'Depth: %{z} m<br>' +
                                                           'Temperature: %{text[0]:.2f} °C<br>' +
                                                           'Salinity: %{text[1]:.2f} psu<br>' +
                                                           'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                           'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                           'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                           'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                           'Label: %{text[6]}<extra></extra>',
                                             text=df_display[
                                                 ["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN", "P_NITRATE", "P_SILICATE",
                                                  "P_PHOSPHATE", column]]
                                             ))

    # Update figure layout
    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]",
                                        xaxis=dict(range=[longitude_min, longitude_max]),
                                        yaxis=dict(range=[latitude_min, latitude_max]),
                                        zaxis=dict(range=[depth_min, depth_max])
                                        ),
                             uirevision=True)

    # Show the plot
    figure_geo.show()


def coupled_label_plot(df, color_label="color", save_dir=None, suffix="", save_as=None, figsize=(6, 6),
                       fontsize=8, ticklabelsize=8,
                       adjust_left=0, adjust_right=0.92, adjust_top=1.1, adjust_bottom=-0.05, pointsize=0.5, dpi=1000,
                       xlabelpad=20, ylabelpad=0, zlabelpad=0):
    """ Plots cluster labels in geographical and embedded space. """
    if save_as is None:
        save_as = ["labels_in_geospace", "labels_in_umapspace"]
    temp = df.copy()

    # Geographic plot
    filename = save_dir + save_as[0] + suffix + ".png" if save_dir is not None else None
    plot_geo(temp, color_label=color_label, save_as=filename, figsize=figsize,
             adjust_left=adjust_left, adjust_right=adjust_right, adjust_top=adjust_top, adjust_bottom=adjust_bottom,
             pointsize=pointsize, dpi=dpi, xlabelpad=xlabelpad, ylabelpad=ylabelpad, zlabelpad=zlabelpad)

    # UMAP plot
    filename = save_dir + save_as[1] + suffix + ".png" if save_dir is not None else None
    plot_embedding(temp, color_label=color_label, alpha=1, save_as=filename, figsize=figsize, fontsize=fontsize,
                   ticklabelsize=ticklabelsize, dpi=dpi)
