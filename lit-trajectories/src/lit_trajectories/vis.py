import altair as alt
import pandas as pd
import lit_trajectories.embedder as embedder


def trimap_scatterplot(chunks):
    """
    Takes a list of embeddings, calculates the trimap for them, then writes out an interactive (html) chart.

    Args:
        embeddings (list): A list of high-dimensional embeddings.

    Returns:
        None. Saves the interactive chart as an HTML file.
    """
    # Calculate trimap
    low_dim_embedding = embedder.trimap_embeddings(chunks)

    # Create DataFrame from input embeddings
    df = pd.DataFrame(chunks, columns=["title", "index", "text"])

    # Add trimap results to the DataFrame
    df["x"] = low_dim_embedding[:, 0]
    df["y"] = low_dim_embedding[:, 1]

    # Create slider for interactivity
    slider = alt.binding_range(min=0, max=df["index"].max(), step=1, name="index")
    cutoff = alt.param(bind=slider, value=0)

    alt.Chart(df).mark_point().encode(
        x="x",
        y="y",
        color=alt.Color("title").legend(None),
        opacity=alt.condition(alt.datum.index == cutoff, alt.value(1), alt.value(0.2)),
        tooltip=["title", "index", "text"],
    ).add_params(cutoff).interactive().save("chart.html")
