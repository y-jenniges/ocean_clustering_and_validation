import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import glasbey


def color_code_labels(df, label_name="label_embedding", color_noise_black=False, drop_noise=False):
    """ Add a color for each label in the clustering using the Glasbey library. """
    temp = df.copy()

    # Define colors
    unique_labels = np.sort(np.unique(temp[label_name]))
    colors = glasbey.create_palette(palette_size=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    temp["color"] = temp[label_name].map(lambda x: color_map[x])

    # How to deal with -1 labels (which is noise in DBSCAN)
    if color_noise_black:
        temp.loc[temp[label_name] == -1, "color"] = "#000000"
    if drop_noise:
        temp = temp[temp[label_name] != -1]

    return temp


# Plot settings
scatter_size = 1.5
score_map = {"Silhouette": "silhouette", "Calinski-Harabasz": "calinski", "Davies-Bouldin": "davies_bouldin",
             "N clusters": "nclusters"}

# Load labels
labels = pd.read_csv("../data/ward_labels.csv")

# Load score data
data = pd.read_csv("../data/ward_scores.csv")
data = data.drop(data[data.n_clusters == 1].index).reset_index(drop=True)
data.distance_threshold = data.distance_threshold.astype(str)
groupby_cols = ['clustering_on', 'scores_on', 'n_clusters', 'distance_threshold', 'linkage']
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # average over iterations
data = data.sort_values(['n_clusters', 'distance_threshold', 'linkage'])
n_clusterss = np.sort(data.n_clusters.unique())  # all n_clusters

print("Data loaded")

# Current hyperparameter values
cur_n_clusters = 2
cur_score = "calinski"

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

    html.Div(dcc.Graph(figure=line_score, id='line-score', clickData={'points': [
        {'x': cur_n_clusters,
         'y': data[data.n_clusters == cur_n_clusters].iloc[0][cur_score],
         'pointNumber': 0}]}
                       ),
             style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Graph(figure=fig_geo, id='fig-geo'),
             style={'margin': dict(l=20, r=20, t=20, b=20),
                    'display': 'inline-block'}),

    html.Div([html.Div([html.Label("Choose which data to cluster:"),
                        dcc.RadioItems(id="data_type",
                                       options=[
                                           {'label': 'Original data', 'value': 'label_original'},
                                           {'label': 'Embedded data', 'value': 'label_embedding'}],
                                       value="label_original",
                                       )
                        ]),
              html.Pre(id="textarea", children="Current parameters: ",
                       style={"white-space": "pre-wrap", "resize": "none", "width": "100%", "height": "auto",
                              "border": "none"}),
              ],
             style={'margin': dict(l=20, r=20, t=20, b=20), "width": "49%", "height": "49%", "resize": "none",
                    'display': 'inline-block'}),

    html.Div(dcc.Graph(figure=fig_umap, id='fig-umap'),
             style={'margin': dict(l=20, r=20, t=20, b=20),
                    'display': 'inline-block'})
])


@app.callback(
    Output('textarea', 'children'),
    Output('line-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),

    Input('score', 'value'),
    Input('line-score', 'clickData'),
    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
    Input('data_type', 'value')
)
def update_heatmap(score, clickData, figure_geo, figure_umap, data_type):
    # Filter data
    print(f"Filter data {data_type}")
    temp = data[(data.clustering_on == data_type.split("_")[1]) & (data.scores_on == data_type.split("_")[1])]
    temp = temp.drop(['clustering_on', 'scores_on'], axis=1)

    # Update score plot
    print(f"Update score plot")
    new_line = px.line(temp, x='n_clusters', y=score_map[score], markers=True)
    new_line.update_traces(marker=dict(color=['red'] * len(n_clusterss)))

    if clickData or data_type:
        print(f"  {data_type}")
        print(f"  {clickData}")

        # Get click coordinates
        x = clickData['points'][0]['x']
        score_value = temp[temp.n_clusters == x][score_map[score]].values[0]
        point_number = clickData['points'][0]['pointNumber']

        # Draw selected point red, all others blue
        trace = next(new_line.select_traces())
        colors = ['blue'] * len(trace.x)
        colors[point_number] = 'red'
        trace.marker.color = colors

        # Update label plots
        print("  Update label plots")
        cur_labels = labels[labels.n_clusters == x]
        cur_labels = color_code_labels(cur_labels, label_name=data_type)

        figure_geo = go.Figure(data=go.Scatter3d(name=f"{x}-geo",
                                                 x=cur_labels.LONGITUDE, y=cur_labels.LATITUDE, z=cur_labels.LEV_M * -1,
                                                 mode='markers',
                                                 marker=dict(size=scatter_size, color=cur_labels.color, opacity=1)))
        figure_umap = go.Figure(data=go.Scatter3d(name=f"{x}-umap",
                                                  x=cur_labels.e0, y=cur_labels.e1, z=cur_labels.e2,
                                                  mode='markers',
                                                  marker=dict(size=scatter_size, color=cur_labels.color, opacity=1)))
        figure_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        figure_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return 'Current parameters: \nn_clusters = {}\n{} = {}'.format(x, score_map[score], np.round(score_value, 2)), \
               new_line, figure_geo, figure_umap
    else:
        return "", new_line, figure_geo, figure_umap


# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
