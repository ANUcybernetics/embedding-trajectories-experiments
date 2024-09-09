import lit_trajectories.texts as texts
import lit_trajectories.vis as vis
import os


def open_file(filename):
    os.system(f"open {filename}")


def main() -> int:
    chunks = texts.aesop_paragraphs()
    # df = vis.trimap_df(chunks)
    df = vis.pacmap_df(chunks)

    trail_chart = vis.trailplot(df)
    trail_chart.save("trail_chart.html")
    open_file("trail_chart.html")

    # scatter_chart = vis.scatterplot(df)
    # scatter_chart.save("scatter_chart.html")
    # open_file("scatter_chart.html")

    return 0
