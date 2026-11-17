
import plotly.graph_objects as go


def create_movement_plot(records, target_word):

    x = [record["x"] for record in records]
    y = [record["y"] for record in records]
    z = [record["z"] for record in records]

    sentences = [
        record["sentence"] for record in records
    ]

    labels = [
        f"Context {record['context_number']}"
        for record in records
    ]

    figure = go.Figure()

    # Draw the movement path
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(
                color="white",
                width=4
            ),
            name="Movement path",
            hoverinfo="skip"
        )
    )

    # Draw the individual contextual representations
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers+text",
            text=labels,
            textposition="top center",
            customdata=sentences,
            marker=dict(
                size=8,
                color=list(range(1, len(records) + 1)),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Context")
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Sentence: %{customdata}<br>"
                "X: %{x:.3f}<br>"
                "Y: %{y:.3f}<br>"
                "Z: %{z:.3f}"
                "<extra></extra>"
            ),
            name="Word representations"
        )
    )

    figure.update_layout(
        title=f"LEXIS | Movement of '{target_word}'",
        template="plotly_dark",
        height=700,
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3"
        )
    )

    return figure
