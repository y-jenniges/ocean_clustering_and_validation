import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
import config

# Open plot in browser
pio.renderers.default = "browser"

# Load an embedding
i = 33
filename = f"{config.output_dir_uncertainty}umap_dbscan_{i}.csv"
df = pd.read_csv(filename)

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=df["e0"], y=df["e1"], z=df["e2"],
    mode='markers',
    marker=dict(size=1, opacity=0.8)  # color=z, colorscale='Viridis',
)])

# Customize layout
fig.update_layout(
    title=f"Embedding: {filename}",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

# Show in browser
fig.show()
