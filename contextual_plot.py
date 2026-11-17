
import plotly.graph_objects as go


def create_contextual_plot(records, target_word):

    x_values = [record["x"] for record in records]
    y_values = [record["y"] for record in records]
    z_values = [record["z"] for record in records]

    context_numbers = [
        record["context_number"]
        for record in records
    ]

    sentences = [
        record["sentence"]
        for record in records
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,

            mode="markers+text",

            text=[
                f"Context {number}"
                for number in context_numbers
            ],

            textposition="top center",

            marker=dict(
                size=8,
                color=context_numbers,
                colorscale="Viridis",
                opacity=0.9,
                showscale=True,
                colorbar=dict(
                    title="Context"
                )
            ),

            customdata=sentences,

            hovertemplate=(
                "<b>%{text}</b><br>"
                "Sentence: %{customdata}<br>"
                "X: %{x:.3f}<br>"
                "Y: %{y:.3f}<br>"
                "Z: %{z:.3f}"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        title=f"Contextual representations of '{target_word}'",

        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3"
        ),

        template="plotly_dark",

        height=700,

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=60
        )
    )

    return figure
