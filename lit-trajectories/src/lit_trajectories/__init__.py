import lit_trajectories.texts as texts
import lit_trajectories.vis as vis
import subprocess


def open_file(filename):
    subprocess.run(["open", filename])


def main() -> int:
    chunks = texts.aesop_paragraphs()
    df = vis.trimap_df(chunks)

    scatter_chart = vis.trimap_scatterplot(df)
    trail_chart = vis.trimap_trailplot(df)

    # Display the charts

    # Or save them to files if needed
    scatter_chart.save("scatter_chart.html")
    trail_chart.save("trail_chart.html")

    open_file("scatter_chart.html")
    open_file("trail_chart.html")

    return 0
