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
    trail_chart.save("output/trail_chart.html")
    open_file("output/trail_chart.html")

    # scatter_chart = vis.scatterplot(df)
    # scatter_chart.save("output/scatter_chart.html")
    # open_file("output/scatter_chart.html")

    return 0
