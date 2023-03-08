import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from dm_control import suite
from IPython.display import HTML


def all_env():
    max_len = max(len(d) for d, _ in suite.BENCHMARKING)
    for domain, task in suite.BENCHMARKING:
        print(f"{domain:<{max_len}}  {task}")


def display_video(video_name, frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=frames,
        interval=interval,
        blit=True,
        repeat=False,
    )
    anim.save(video_name)
    return HTML(anim.to_html5_video())
