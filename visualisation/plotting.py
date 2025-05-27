import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import glasbey
import gsw
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from shapely.geometry import box
from cartopy.io import shapereader


def plot_each_depth_level(df, color_label="water", save_as="output/grid_cells.png"):
    """ Scatter plot for each depth level with a selectable color column. """
    # Show grid cells for every depth layer
    depths = df["LEV_M"].unique()
    lamin = df["LATITUDE"].min() - 5
    lamax = df["LATITUDE"].max() + 5
    lomin = df["LONGITUDE"].min() - 5
    lomax = df["LONGITUDE"].max() + 5

    # Create figure
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(2 * 6, 10), sharex=True, sharey=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()

    # Plot every depth level
    for i in range(len(depths)):
        d = depths[i]
        sel = df[df["LEV_M"] == d]

        axs[i].coastlines()
        axs[i].gridlines(draw_labels=True)
        axs[i].set_extent([lomin, lomax, lamin, lamax], crs=ccrs.PlateCarree())
        axs[i].scatter(sel["LONGITUDE"], sel["LATITUDE"], s=0.5, c=sel[color_label])
        axs[i].set_title(f"{d}m")

    # Show and save
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def plot_embedding(embedding, color_label=None, alpha=0.08, size=2, save_as=None, figsize=(6, 6), fontsize=8,
                   ticklabelsize=8, dpi=1000):
    """ Plot a 2d or 3d embedding. (It can be a pandas.DataFrame or a numpy.array.) """
    # Determine x, y, z, data and colour
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

    # Define figure
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
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.tick_params(axis='y', labelsize=ticklabelsize)
    ax.tick_params(axis='z', labelsize=ticklabelsize)
    plt.tight_layout()
    plt.subplots_adjust(left=-0.01, right=0.92, top=1.1, bottom=0)
    if save_as:
        plt.savefig(save_as, dpi=dpi)
    plt.show()


def plot_geo(df, color_label="color", save_as=None, figsize=(6, 6),
             adjust_left=0, adjust_right=0.92, adjust_top=1.1, adjust_bottom=-0.05, pointsize=0.5, dpi=600,
             xlabelpad=20, ylabelpad=0, zlabelpad=0):
    """ 3d scatter plot of a cluster set with given colors. """

    # Define figure and plot grid cells as scatter points
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df["LONGITUDE"], df["LATITUDE"], df["LEV_M"], c=df[color_label], s=pointsize, alpha=1, zorder=1)

    # Get coastlines from Cartopy feature
    # Create a bounding box for the data region
    bbox = box(df["LONGITUDE"].min(), df["LATITUDE"].min(), df["LONGITUDE"].max(), df["LATITUDE"].max())

    # Get cartopy coastline
    shpfilename = shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
    reader = shapereader.Reader(shpfilename)

    # Loop through geometries and clip to desired range
    for record in reader.records():
        # Clip geometry to desited range
        geom = record.geometry.intersection(bbox)

        # Skip geometry, if no intersection with desired range
        if geom.is_empty:
            continue

        # Convert geometry to a list of lines
        if geom.geom_type == "MultiLineString":
            lines = geom.geoms
        elif geom.geom_type == "LineString":
            lines = [geom]
        else:
            continue  # skip if it's not a line

        # Add each line to the plot
        for line in lines:
            x, y = line.xy
            z = np.full_like(x, 0.0)  # Place at surface
            ax.plot(x, y, z, color='black', linewidth=1.5, zorder=10)

    # Set axis limits
    ax.set_xlim(df["LONGITUDE"].min(), df["LONGITUDE"].max())
    ax.set_ylim(df["LATITUDE"].min(), df["LATITUDE"].max())

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


def plot_interactive_geo(df, column="label", color_label="color", scatter_size=3, margin=5):
    """ Interactive 3d geographic scatter plot. """
    df_display = df.copy()

    longitude_min = df["LONGITUDE"].min()
    longitude_max = df["LONGITUDE"].max()
    latitude_min = df["LATITUDE"].min()
    latitude_max = df["LATITUDE"].max()
    depth_min = df["LEV_M"].min()
    depth_max = df["LEV_M"].max()

    # Define figure
    figure_geo = go.Figure(data=go.Scatter3d(x=df_display.LONGITUDE, y=df_display.LATITUDE, z=df_display.LEV_M,
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
                                        zaxis=dict(range=[depth_max, depth_min])
                                        ),
                             uirevision=True)

    # Show the plot
    figure_geo.show()


def plot_interactive_embedding(df, color_label=None, scatter_size=1, save_as=None):
    # Define plot parameters
    if color_label:
        plot_params = {"x": df["e0"], "y": df["e1"], "z": df["e2"], "mode": "markers",
                       "marker": dict(size=scatter_size, opacity=0.8), "color": df[color_label]}

    else:
        plot_params = {"x": df["e0"], "y": df["e1"], "z": df["e2"], "mode": "markers",
                       "marker": dict(size=scatter_size, opacity=0.8)}

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(**plot_params)])

    # Customize layout
    fig.update_layout(
        title=f"Embedding",
        scene=dict(
            xaxis_title="X-Axis",
            yaxis_title="Y-Axis",
            zaxis_title="Z-Axis"
        )
    )

    # Save
    if save_as:
        fig.write_html(save_as)

    # Show in browser
    fig.show()


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


def plot_ts(df, figsize=(4, 4), dpi=None, ncols=5, xlim=None, ylim=None,
            save_as=None, fontsize=None,
            adjust_left=0, adjust_right=1, adjust_top=1, adjust_bottom=0, legend_loc="center right",
            anchor=(0.0, -0.15)):
    """ Plot TS diagram. """
    temp = df.copy()

    # Compute necessary parameters
    temp["pressure"] = gsw.p_from_z(-1 * temp["LEV_M"], temp["LATITUDE"])
    temp["abs_salinity"] = gsw.SA_from_SP(temp["P_SALINITY"], temp["pressure"], temp["LONGITUDE"], temp["LATITUDE"])
    temp["cons_temperature"] = gsw.CT_from_pt(temp["abs_salinity"], temp["P_TEMPERATURE"])
    temp["rho"] = gsw.rho(temp["abs_salinity"], temp["cons_temperature"], temp["pressure"])

    # Plot limits
    smin = temp["abs_salinity"].min() - (0.01 * temp["abs_salinity"].min())
    smax = temp["abs_salinity"].max() + (0.01 * temp["abs_salinity"].max())
    tmin = temp["cons_temperature"].min() - (0.1 * temp["cons_temperature"].max())
    tmax = temp["cons_temperature"].max() + (0.1 * temp["cons_temperature"].max())

    if xlim:
        smin = xlim[0] - (0.01 * xlim[0])
        smax = xlim[1] + (0.01 * xlim[1])

    if ylim:
        tmin = ylim[0] - (0.01 * ylim[0])
        tmax = ylim[1] + (0.01 * ylim[1])

    # Number of gridcells in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1, 0))
    ydim = int(round((tmax - tmin) / 0.1 + 1, 0))

    # Empty grid
    dens = np.zeros((ydim, xdim))

    # Temperature and salinity vectors
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    ti = np.linspace(1, ydim - 1, ydim) * 0.1 + tmin

    # Fill grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # Convert to sigma-t
    dens = dens - 1000

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    contours = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(contours, fontsize=fontsize, inline=1, fmt='%1.1f')  # label every second level
    for cluster in np.sort(temp["label"].unique()):
        cluster_points = temp[temp["label"] == cluster]
        ax.scatter(x=cluster_points["abs_salinity"], y=cluster_points["cons_temperature"], c=cluster_points["color"],
                   label=cluster, s=9, alpha=1, marker=".")
    ax.set_xlabel('Absolute salinity [g/kg]', fontsize=fontsize)
    ax.set_ylabel('Conservative temperature [°C]', fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.subplots_adjust(left=adjust_left, right=adjust_right, top=adjust_top, bottom=adjust_bottom)
    if ncols:
        ax.legend(loc=legend_loc, title='Clusters', bbox_to_anchor=anchor, ncol=ncols, markerscale=5, frameon=False,
                  handletextpad=0.1)

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=dpi)
    plt.show()


def compare_stats(df, labels, vars=None, vars_map=None, save_as=None, sort_labels=True, dpi=600, figsize=(8, 6)):
    """ Compare per-parameter-statistics of multiple labels of a clustering. """
    if not vars:
        vars = np.sort(['P_TEMPERATURE', 'P_SALINITY', 'P_OXYGEN', 'P_NITRATE', 'P_SILICATE', 'P_PHOSPHATE'])
        vars_map = {"P_TEMPERATURE": "Temperature", "P_SALINITY": "Salinity", "P_OXYGEN": "Oxygen",
                    "P_NITRATE": "Nitrate", "P_SILICATE": "Silicate", "P_PHOSPHATE": "Phosphate"}

    temp = df[df.label.isin(labels)]  # Filter for the given interesting regions
    scaler = MinMaxScaler().fit(temp[vars])  # Define scaler for the regions
    temp_s = pd.DataFrame(scaler.transform(temp[vars]), columns=vars, index=temp.index)  # Scale data for comparability
    temp_s["label"] = temp["label"]  # Adding label information
    temp_m = pd.melt(temp_s, id_vars=["label"], value_vars=vars)  # Wide to long format
    if vars_map:
        temp_m.variable = temp_m.variable.map(vars_map)  # Renaming

    # Define colors according to original ones
    my_pal = {}
    for label in labels:
        my_pal[label] = df[df.label == label].iloc[0].color

    # Define sequence of labels
    if sort_labels:
        labels = np.sort(labels)

    # Rename axis so it looks nice on the legend
    temp_m = temp_m.rename(columns={"label": "Clusters"})

    # plot
    plt.figure(figsize=figsize)
    bp = sns.boxplot(temp_m, x="variable", y="value", hue="Clusters", palette=my_pal, flierprops={"marker": "."},
                     hue_order=labels)
    sns.move_legend(bp, "lower left")
    plt.xlabel("")
    plt.ylabel("Scaled value")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=dpi)
    plt.show()


def plot_sankey(df, source_col="label", target_col="emu_label",
                source_name="Our Cluster Set", target_name="EMU",
                figsize=(10, 6), save_as=None, dpi=100):
    # Convert labels to string (-> avoid integer issues)
    df[source_col] = df[source_col].astype(str)
    df[target_col] = df[target_col].astype(str)

    # Create all unique labels and index mapping
    all_labels = pd.Series(pd.concat([df[source_col], df[target_col]])).unique()
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Count flows
    flow_counts = df.groupby([source_col, target_col]).size().reset_index(name="count")
    flow_counts["source_idx"] = flow_counts[source_col].map(label_to_index)
    flow_counts["target_idx"] = flow_counts[target_col].map(label_to_index)

    # Convert figsize to pixels
    width_px = int(figsize[0] * 96)
    height_px = int(figsize[1] * 96)

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=list(label_to_index.keys()),
            color="blue"
        ),
        link=dict(
            source=flow_counts["source_idx"],
            target=flow_counts["target_idx"],
            value=flow_counts["count"]
        )
    )])

    fig.update_layout(
        title_text=f"Sankey Diagram: {source_name} - {target_name}",
        font_size=10,
        width=width_px,
        height=height_px
    )
    fig.show()

    # Optionally save diagram
    if save_as:
        fig.write_image(save_as, scale=dpi / 100)
