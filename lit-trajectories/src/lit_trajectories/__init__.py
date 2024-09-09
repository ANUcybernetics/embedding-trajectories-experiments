import lit_trajectories.texts as texts
import lit_trajectories.vis as vis
import os


def open_file(filename):
    os.system(f"open {filename}")


def main() -> int:
    chunks = texts.aesop_paragraphs()
    df = vis.trimap_df(chunks)

    trail_chart = vis.trimap_trailplot(df)
    trail_chart.save("trail_chart.html")
    open_file("trail_chart.html")

    # scatter_chart = vis.trimap_scatterplot(df)
    # scatter_chart.save("scatter_chart.html")
    # open_file("scatter_chart.html")

    return 0
