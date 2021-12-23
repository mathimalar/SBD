import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib
import SBD
import matplotlib.colors as colors


def plot_dos(dos) -> None:
    """Plots a 3D matrix with a slider for dim1."""
    matplotlib.use('TkAgg')
    init_idx = 0
    fig, ax = plt.subplots()
    img = plt.imshow(dos[init_idx], cmap='hot')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.1, bottom=0.25)
    ax_energy = plt.axes([0.25, 0.1, 0.65, 0.03])

    def update(val):
        # index_slider.val = round(index_slider.val)
        img.set_data(dos[index_slider.val])
        img.set_norm(matplotlib.colors.Normalize())
        index_slider.valtext.set_text(index_slider.val)
        fig.canvas.draw_idle()

    index_slider = Slider(
        ax=ax_energy,
        label='Energy index',
        valmin=0,
        valmax=np.shape(dos)[0] - 1,
        valinit=init_idx,
        valstep=1
    )

    index_slider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        index_slider.reset()

    button.on_clicked(reset)

    plt.show()


def plot_fft(A, X, Y, init_idx: int = 0) -> None:
    """
    Plotting the measurement (big), the kernel and activation maps, and the measurements FFT along with the kernel FFT.
    """
    max_idx = np.shape(Y)[0] - 1
    assert init_idx in range(max_idx), 'The initial value is out of bounds.'

    matplotlib.use('TkAgg')

    A_fft = np.fft.fftshift(np.fft.fft2(A, s=np.shape(Y)[1:]))
    Y_fft = np.fft.fftshift(np.fft.fft2(SBD.normalize_measurement(Y), s=np.shape(Y)[1:]))

    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(2, 4, wspace=0.2, hspace=0.3)

    Y_plot = fig.add_subplot(grid[:, :2])
    Y_plot.set_title(f'Measurement at slice: {init_idx}')
    Y_plot.axis('off')

    Y_fft_plot = fig.add_subplot(grid[1, 2])
    Y_fft_plot.set_title('FFT(Y) Abs')
    Y_fft_plot.get_yaxis().set_visible(False)
    Y_fft_plot.get_xaxis().set_visible(False)

    A_fft_plot = fig.add_subplot(grid[1, 3])
    A_fft_plot.set_title('Kernel FFT')
    A_fft_plot.get_yaxis().set_visible(False)
    A_fft_plot.get_xaxis().set_visible(False)

    A_plot = fig.add_subplot(grid[0, 2])
    A_plot.set_title('Recovered kernel')
    A_plot.get_yaxis().set_visible(False)
    A_plot.get_xaxis().set_visible(False)

    X_plot = fig.add_subplot(grid[0, 3])
    X_plot.set_title('Activation map')
    X_plot.get_yaxis().set_visible(False)
    X_plot.get_xaxis().set_visible(False)

    Y_im = Y_plot.imshow(Y[init_idx], cmap='hot')

    Y_fft_im = Y_fft_plot.imshow(np.abs(Y_fft[init_idx]), cmap='hot')  # Should be logarithmic?
    A_fft_im = A_fft_plot.imshow(np.abs(A_fft[init_idx]), cmap='hot')  # Should be logarithmic?
    A_plot_im = A_plot.imshow(A[init_idx], cmap='hot')
    X_plot_im = X_plot.imshow(X.A == 0, cmap='binary', interpolation='None')

    plt.colorbar(A_fft_im, ax=A_fft_plot)
    plt.colorbar(Y_fft_im, ax=Y_fft_plot)

    ax_energy = plt.axes([0.25, 0.01, 0.65, 0.03])
    index_slider = Slider(
        ax=ax_energy,
        label='Energy index',
        valmin=0,
        valmax=max_idx,
        valinit=init_idx,
        valstep=1
    )

    def update(val):
        # index_slider.val = round(index_slider.val)
        Y_im.set_data(Y[index_slider.val])
        Y_im.set_norm(matplotlib.colors.Normalize())
        Y_plot.set_title(f'Measurement at slice: {index_slider.val}')

        Y_fft_im.set_data((np.abs(Y_fft[index_slider.val])))
        Y_fft_im.set_norm(matplotlib.colors.Normalize())

        A_fft_im.set_data((np.abs(A_fft[index_slider.val])))
        A_fft_im.set_norm(matplotlib.colors.Normalize())

        A_plot_im.set_data(A[index_slider.val])
        A_plot_im.set_norm(matplotlib.colors.Normalize())
        index_slider.valtext.set_text(index_slider.val)
        fig.canvas.draw_idle()

    index_slider.on_changed(update)
    # plt.tight_layout()
    plt.show()


def plot_benchmark(e_matrix, d_range, k_range) -> None:
    D, K = np.meshgrid(d_range, k_range / 256)
    im = plt.pcolormesh(D, K, e_matrix, cmap='jet', shading='auto')
    cb = plt.colorbar()
    cb.set_label(r'$\epsilon$', loc='bottom')
    plt.xlabel('Defect density')
    plt.ylabel(r'Kernel relative size $m/n$')
    plt.clim(0, 1)
    plt.xscale('log')
    plt.savefig('benchmark.jpg')
    plt.show()


def side_by_side(M1, M2, figtitle, ax1_title, ax2_title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(figtitle)
    ax[0].imshow(M1, cmap='hot')
    ax[0].set_title(ax1_title)
    ax[1].imshow(M2, cmap='hot')
    ax[1].set_title(ax2_title)
    for i in range(2):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def illustrate(A_true, X_true, Y, A_solved, X_solved) -> None:
    """
    Plots a comparison figure for a simulated measurement
    """
    A_true_fft = np.fft.fftshift(np.fft.fft2(A_true, s=np.shape(Y)))
    A_solved_fft = np.fft.fftshift(np.fft.fft2(A_solved, s=np.shape(Y)))
    Y_fft = np.fft.fftshift(np.fft.fft2(Y, s=np.shape(Y)))

    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(3, 8, wspace=0.2, hspace=0.3)

    Y_plot = fig.add_subplot(grid[:-1, 2:-2])
    Y_fft_plot = fig.add_subplot(grid[-1, 4:6])

    A_true_plot = fig.add_subplot(grid[0, :2])
    X_true_plot = fig.add_subplot(grid[1, :2])
    A_true_fft_plot = fig.add_subplot(grid[2, :2])

    A_solved_plot = fig.add_subplot(grid[0, -2:])
    X_solved_plot = fig.add_subplot(grid[1, -2:])
    A_solved_fft_plot = fig.add_subplot(grid[2, -2:])

    Y_plot.imshow(Y, cmap='hot')
    Y_fft_plot.imshow(np.real(Y_fft), cmap='bwr', norm=colors.CenteredNorm())

    A_true_plot.imshow(A_true, cmap='hot')
    X_true_plot.imshow(X_true.A != 0, cmap='hot')
    A_true_fft_plot.imshow(np.real(A_true_fft), cmap='bwr', norm=colors.CenteredNorm())

    A_solved_plot.imshow(A_solved, cmap='hot')
    X_solved_plot.imshow(X_solved != 0, cmap='hot')
    A_solved_fft_plot.imshow(np.real(A_solved_fft), cmap='bwr', norm=colors.CenteredNorm())

    plt.show()
