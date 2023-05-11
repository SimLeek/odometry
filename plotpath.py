import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
from abc import abstractmethod


class PathPlotter3D(object):
    def __init__(self, start_pos=(0, 0, 0), color="green", fps=30, title="3D path"):
        self.z = np.asarray([start_pos[0]])
        self.x = np.asarray([start_pos[1]])
        self.y = np.asarray([start_pos[2]])
        self.color = color

        self.fig = plt.figure()

        self.ax = plt.axes(projection="3d")
        self.ax.set_title(title)

        self.plot = self.ax.plot3D(self.x, self.y, self.z, self.color)

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=None,
            init_func=self.init_plot,
            blit=False,
            interval=1000.0 / fps,
        )
        plt.show()

    def init_plot(self):
        return (self.plot,)

    def append_xyz(self, x, y, z):
        assert x.shape == y.shape == z.shape
        assert (
            len(x.shape)
            == len(y.shape)
            == len(z.shape)
            == len(self.x.shape)
            == len(self.y.shape)
            == len(self.z.shape)
            == 1
        )
        self.x = np.concatenate((self.x, x))
        self.y = np.concatenate((self.y, y))
        self.z = np.concatenate((self.z, z))

    @abstractmethod
    def update_callback(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return None

    def update(self, frame_number):
        new_xyz = self.update_callback()
        if new_xyz is not None:
            x, y, z = new_xyz
            self.append_xyz(np.asarray(x), np.asarray(y), np.asarray(z))

        self.plot[0].remove()
        (self.plot[0],) = self.ax.plot3D(self.x, self.y, self.z, color=self.color)


class LinearPlot3D(PathPlotter3D):
    def update_callback(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return (
            np.asarray([self.x[-1] + 1]),
            np.asarray([self.y[-1] + 1]),
            np.asarray([self.z[-1] + 1]),
        )


LinearPlot3D()
