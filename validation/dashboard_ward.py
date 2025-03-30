import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd

import config
from visualisation.plotting import color_code_labels
from utils.analysis import prepare_labels_df


# Plot settings
scatter_size = 1.5
score_map = {"Silhouette": "silhouette", "Calinski-Harabasz": "calinski_harabasz", "Davies-Bouldin": "davies_bouldin",
             "N clusters": "nclusters"}

# Load original data
df_original = pd.read_csv(f"{config.output_dir}/wide_table_knn.csv")

# Load labels
iteration = 0
labels = pd.read_csv(f"{config.output_dir_clustering}/labels_ward.csv")
labels = prepare_labels_df(labels, iteration=iteration)

# Score data
data = pd.read_csv(f"{config.output_dir_clustering}internal_validation_ward.csv")
data = data.drop(["clustering"], axis=1)
data["distance_threshold"] = data.distance_threshold.astype(str)
groupby_cols = ["preprocessing"] + list(config.algorithms_and_hyps["ward"][1].keys())
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # Average over iterations
data = data.sort_values(list(config.algorithms_and_hyps["ward"][1].keys()))
n_clusterss = np.sort(data["n_clusters"].unique())  # All n_clusters

print("Data loaded")

# Current hyperparameter values
cur_n_clusters = 2
cur_score = "calinski_harabasz"

# Figures
fig_geo = go.Figure()
fig_umap = go.Figure()
line_score = px.line(data, x='n_clusters', y=cur_score, markers=True)

print("Figures defined")


# Dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div([dcc.RadioItems(['Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette'], 'Calinski-Harabasz',
                             id='score', labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
              ]),

    html.Div(dcc.Graph(figure=line_score, id='line-score',
                       clickData={'points': [{'x': cur_n_clusters,
                                              'y': data[data.n_clusters == cur_n_clusters].iloc[0][cur_score],
                                              'pointNumber':  0}]}),
             style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Graph(figure=fig_geo, id='fig-geo'),
             style={'margin': dict(l=20, r=20, t=20, b=20), 'display': 'inline-block'}),

    html.Div(
        dcc.RadioItems(id="selection-state", value='select all',
                       options=['select all', 'select', 'deselect'], labelStyle={'display': 'inline-block'})
    ),

    html.Div([html.Div([html.Label("Preprocessing:"),
                        dcc.RadioItems(id="data_type",
                                       options=[
                                           {'label': 'MinMax', 'value': 'minmax'},
                                           {'label': 'MinMax-UMAP', 'value': 'minmax_umap'}],
                                       value="minmax",
                                       )
                        ]),
              html.Pre(id="textarea", children="Current parameters: ",
                       style={"white-space": "pre-wrap", "resize": "none", "width": "100%", "height": "auto",
                              "border": "none"}),
              ],
             style={'margin': dict(l=20, r=20, t=20, b=20), "width": "49%", "height": "49%", "resize": "none",
                    'display': 'inline-block'}),

    html.Div(dcc.Graph(figure=fig_umap, id='fig-umap',
                       clickData={'points': [{'x': None, 'y': None, 'z': None, 'text': None}]}),
             style={'margin': dict(l=20, r=20, t=20, b=20),
                    'display': 'inline-block'}),

    dcc.Store(id="current", data={"label_selection": [], "umap_clickData": None}),
])


@app.callback(
    Output('textarea', 'children'),
    Output('line-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('current', 'data'),

    Input('score', 'value'),
    Input('line-score', 'clickData'),
    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
    Input('fig-umap', 'clickData'),
    Input('selection-state', 'value'),
    Input('current', 'data'),
    Input('data_type', 'value')
)
def update_heatmap(score, clickData, figure_geo, figure_umap, umap_clickData, selection_state, old_params, data_type):
    # Filter data
    print(f"Filter data {data_type}")
    temp = data[data.preprocessing == data_type]

    # Update heatmap
    print(f"Update score plot")
    new_line = px.line(temp, x='n_clusters', y=score_map[score], markers=True)
    new_line.update_traces(marker=dict(color=['red'] * len(n_clusterss)))

    # Update label selection
    prev_label_selection = old_params["label_selection"]
    prev_umap_clickData = old_params["umap_clickData"]
    new_label_selection = []
    if selection_state == 'select':
        if umap_clickData and (prev_umap_clickData != umap_clickData):
            print("  Select")
            selected_label = umap_clickData["points"][0]["text"]
            new_label_selection = prev_label_selection + [selected_label]
    elif selection_state == 'deselect':
        if umap_clickData and (prev_umap_clickData != umap_clickData):
            print("  Deselect")
            selected_label = umap_clickData["points"][0]["text"]
            new_label_selection = [x for x in prev_label_selection if x != selected_label]
    print(f"  New label selection: {new_label_selection}")

    if clickData:
        print(f"  {data_type}")
        print(f"  {clickData}")

        # Get click coordinates
        x = clickData["points"][0]['x']
        score_value = data[data.n_clusters == x][score_map[score]].values[0]
        point_number = clickData["points"][0]['pointNumber']

        # Draw selected point red, all others blue
        trace = next(new_line.select_traces())
        colors = ["blue"] * len(trace.x)
        colors[point_number] = "red"
        trace.marker.color = colors

        # Update label plots
        print("  Update label plots")
        cur_labels = labels[labels["n_clusters"] == x]
        cur_labels = color_code_labels(cur_labels, column_name=data_type)

        geo_labels = cur_labels.copy()
        if new_label_selection:
            geo_labels = geo_labels[cur_labels[data_type].isin(new_label_selection)]

        figure_geo = go.Figure(data=go.Scatter3d(name=f"{x}-geo",
                                                 x=geo_labels.LONGITUDE, y=geo_labels.LATITUDE, z=geo_labels.LEV_M * -1,
                                                 mode='markers',
                                                 marker=dict(size=scatter_size, color=geo_labels.color, opacity=1),
                                                 hovertemplate='Longitude: %{x}<br>' +
                                                               'Latitude: %{y}<br>' +
                                                               'Depth: %{z}<br>' +
                                                               'Label: %{text}<extra></extra>',
                                                 text=geo_labels[data_type]
                                                 ))
        figure_umap = go.Figure(data=go.Scatter3d(name=f"{x}-umap",
                                                  x=cur_labels.e0, y=cur_labels.e1, z=cur_labels.e2,
                                                  mode='markers',
                                                  marker=dict(size=scatter_size, color=cur_labels.color, opacity=1),
                                                  hovertemplate='x: %{x}<br>' +
                                                                'y: %{y}<br>' +
                                                                'z: %{z}<br>' +
                                                                'Label: %{text}<extra></extra>',
                                                  text=cur_labels[data_type]
                                                  ))
        figure_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        figure_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return 'Current parameters: \nn_clusters = {}\n{} = {}'.format(x, score_map[score], np.round(score_value, 2)), \
               new_line, figure_geo, figure_umap, \
               {"label_selection": new_label_selection, "umap_clickData": umap_clickData}
    else:
        return "", new_line, figure_geo, figure_umap, {"label_selection": [], "umap_clickData": None}


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
