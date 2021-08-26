"""
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
# from .plotting import jakeStyle
# plt.style.use(jakeStyle)


def FWHM_old(y_values, x_values=None):
    """
    Calculate the FWHM of a series of numerical values on a constant x-grid
    """

    norm = y_values / np.nanmax(y_values)
    half = abs(norm - max(norm) / 2) # Create zero-crossings at half max value
    zero_crossings = []
    zero_crossings.append(np.argmin(half[:len(half)//2])) # First zero-crossing
    zero_crossings.append(np.argmin(half[len(half)//2:]) + len(half)//2) # Second zero-crossing

    fwhm_steps = np.diff(zero_crossings)[0]

    if x_values is not None:
        x_step = np.diff(x_values)[0]
        return fwhm_steps * x_step
    else:
        return fwhm_steps



def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def FWHM(x_values, y_values):
    """
    """

    half_max = max(y_values) / 2
    signs = np.sign(np.add(y_values, -half_max))
    zero_crossings = (signs[0:-2]) != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x_values, y_values, zero_crossings_i[0], half_max),
            lin_interp(x_values, y_values, zero_crossings_i[1], half_max)]



def Gaussian(x, x0=0, sigma=1):
    """
    Generate a simple Gaussian distribution
    """
    
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(1/2) * ((x - x0) / sigma) ** 2)




def animate_results(results=None, vmin=0, vmax=1, cmap="jet", title="", save=False):
    """
    """

    fig = plt.figure()
    ax = fig.gca()

    ims = []
    for r in results:
        im = ax.imshow(r, animated=True, vmin=vmin, vmax=vmax, cmap=cmap)

        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=60, repeat_delay=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        # Set up formatting for the movie files
        writer = animation.writers["ffmpeg"]

        ani.save(f"{title}_animation.mp4", writer=writer(fps=30))
    else:
        plt.show()




def test():
    
    xs = np.linspace(-10, 10, 200)
    dist = Gaussian(xs)
    dist_fwhm = FWHM_old(dist, x_values=xs)
    print(f"FWHM = {dist_fwhm}")


    alt_fwhm = FWHM(xs, dist)
    print(np.diff(alt_fwhm)[0])


if __name__ == "__main__":

    test()
