import altair as alt
import pandas as pd
import numpy as np
import trimap


def calculate_trimap(data):
    low_dim_embedding = trimap.TRIMAP().fit_transform(data)
    print(low_dim_embedding[:10])


def scatterplot():
    rand = np.random.RandomState(42)

    df = pd.DataFrame({"xval": range(100), "yval": rand.randn(100).cumsum()})

    slider = alt.binding_range(min=0, max=100, step=1)
    cutoff = alt.param(bind=slider, value=50)

    alt.Chart(df).mark_point().encode(
        x="xval",
        y="yval",
        color=alt.condition(
            alt.datum.xval < cutoff, alt.value("red"), alt.value("blue")
        ),
    ).add_params(cutoff).save("chart.html")
