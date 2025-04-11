import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

import config


# Open plot in browser
pio.renderers.default = "browser"

# Load an embedding
filename = f"{config.output_dir_nemi}volume_nemi_iteration84.csv"
df = pd.read_csv(filename)
df = df[df.label != 0]

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=df["e0"], y=df["e1"], z=df["e2"],
    mode='markers',
    marker=dict(size=1, opacity=0.8, color=df["label_color"])
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
