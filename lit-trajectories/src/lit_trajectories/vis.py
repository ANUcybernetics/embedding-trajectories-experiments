import altair as alt
import pandas as pd
import lit_trajectories.embedder as embedder


def trimap_df(chunks):
    """
    Takes a list of embeddings, calculates the trimap for them, and returns a DataFrame.

    Args:
        chunks (list): A list of chunks containing title, index, and text.

    Returns:
        pandas.DataFrame: A DataFrame with trimap results added.
    """
    # Calculate trimap
    low_dim_embedding = embedder.trimap_embeddings(chunks)

    # Create DataFrame from input embeddings
    df = pd.DataFrame(chunks, columns=["title", "index", "text"])

    # Add trimap results to the DataFrame
    df["x"] = low_dim_embedding[:, 0]
    df["y"] = low_dim_embedding[:, 1]

    return df


def pacmap_df(chunks):
    """
    Takes a list of embeddings, calculates the pacmap for them, and returns a DataFrame.

    Args:
        chunks (list): A list of chunks containing title, index, and text.

    Returns:
        pandas.DataFrame: A DataFrame with pacmap results added.
    """
    # Calculate pacmap
    low_dim_embedding = embedder.pacmap_embeddings(chunks)

    # Create DataFrame from input embeddings
    df = pd.DataFrame(chunks, columns=["title", "index", "text"])

    # Add pacmap results to the DataFrame
    df["x"] = low_dim_embedding[:, 0]
    df["y"] = low_dim_embedding[:, 1]

    return df


def scatterplot(df):
    # Create slider for interactivity
    slider = alt.binding_range(min=0, max=df["index"].max(), step=1, name="index")
    cutoff = alt.param(bind=slider, value=0)

    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=alt.Color("title").legend(None),
            opacity=alt.condition(
                alt.datum.index == cutoff, alt.value(1), alt.value(0.1)
            ),
            tooltip=["title", "index", "text"],
        )
        .add_params(cutoff)
        .interactive()
    )

    return chart


def trailplot(df):
    # Hover selection
    hover = alt.selection_point(on="mouseover", fields=["title"], empty=False)
    hover_point_opacity = alt.selection_point(on="mouseover", fields=["title"])

    # Search box for title
    search_box = alt.param(
        value="", bind=alt.binding(input="search", placeholder="Title", name="Search ")
    )

    # Base chart
    base = alt.Chart(df).encode(
        x=alt.X("x:Q", scale=alt.Scale(zero=False), title="TriMap X"),
        y=alt.Y("y:Q", scale=alt.Scale(zero=False), title="TriMap Y"),
        color=alt.Color("title:N", legend=None),
        detail="title:N",
    )

    # Points that are always visible (filtered by slider and search)
    visible_points = (
        base.mark_circle(size=60)
        .encode(
            opacity=alt.condition(
                hover_point_opacity
                & alt.expr.test(alt.expr.regexp(search_box, "i"), alt.datum.title),
                alt.value(0.8),
                alt.value(0.1),
            ),
            tooltip=[
                alt.Tooltip("title:N", title="Title"),
                alt.Tooltip("text:N", title="Text"),
            ],
        )
        .add_params(hover, hover_point_opacity)
    )

    hover_line = alt.layer(
        # Line layer
        base.mark_trail().encode(
            order=alt.Order("index:Q", sort="ascending"),
            size=alt.Size(
                "index:Q",
                scale=alt.Scale(
                    domain=[df["index"].min(), df["index"].max()], range=[8, 1]
                ),
                legend=None,
            ),
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            color=alt.value("#222222"),
        ),
        # Point layer
        base.mark_point(size=30).encode(
            opacity=alt.condition(hover, alt.value(0.8), alt.value(0)),
        ),
    )

    # Index labels
    index_labels = (
        base.mark_text(align="left", dx=5, dy=-5, fontSize=12)
        .encode(text="index:O", color=alt.value("#222222"))
        .transform_filter(hover)
    )

    # Title labels
    title_labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=-15, dy=-15, fontSize=14, fontWeight="bold")
        .encode(
            x="x:Q",
            y="y:Q",
            text="title:N",
            color=alt.value("black"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        )
        .transform_filter(alt.datum.index == df["index"].max())
    )

    # Combine all layers
    chart = (
        alt.layer(visible_points, index_labels, title_labels, hover_line)
        .properties(width=700, height=500, padding=10)
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .add_params(search_box)
        .interactive()
    )

    return chart
