import numpy as np

class GradientUtils:
    """
    Class to compute tiling statistics such as gradient histograms and peakiness scores.
    """

    def __init__(self, imgs: np.ndarray, tile_size: int = 32, border_size: int = None, bin_edges: np.ndarray = None):
        """
        Initialize the GradientUtils class.

        Args:
            imgs (numpy array with dims BYXC): a batch of images.
            tile_size (int, optional): size of tiles in X and Y direction. Defaults to 32.
            border_size (int, optional): border width to ignore. Defaults to tile_size // 2.
            bin_edges (numpy array, optional): bin edges for histograms. If None, computed from gradients.
        """
        self.imgs = imgs
        self.tile_size = tile_size
        self.border_size = border_size if border_size is not None else tile_size // 2

        # remove borders
        self.imgs_without_borders = GradientUtils.border_free(self.imgs, self.border_size)

        # gradients
        self.gradients_x, self.gradients_y = GradientUtils.compute_gradients(self.imgs, self.border_size)

        # gradients along tile grid
        self.gradients_edges = self._gradients_along_tile_grid(offset=self.tile_size - 1)
        self.gradients_middle = self._gradients_along_tile_grid(offset=self.tile_size // 2 - 1)

        # compute bin edges (if not given)
        self._bin_edges = bin_edges
        if self._bin_edges is None:
            self._bin_edges = GradientUtils.get_bin_edges(
                [self.gradients_x, self.gradients_y, self.gradients_edges, self.gradients_middle],
                num_bins=200
            )
        # histograms
    @staticmethod
    def compute_histograms(gradients: np.ndarray,bin_edges: np.ndarray):
        """
        Compute histogramsgradients.
        Args:
            gradients(numpy array): gradients
            bin_edges (numpy array): edges of the histogram bins.
        Returns:
            histograms (tuple): histograms for edges and middle gradients.
        """
        return np.histogram(gradients, bins=bin_edges)[0]

    @staticmethod
    def compute_gradients(imgs: np.ndarray, border_size: int = 0):
        """Compute horizontal and vertical gradients for an image batch."""
        wb = GradientUtils.border_free(imgs, border_size)
        grad_x = wb[:, :, 1:, :] - wb[:, :, :-1, :]  # horizontal
        grad_y = wb[:, 1:, :, :] - wb[:, :-1, :, :]  # vertical
        return grad_x, grad_y

    @staticmethod
    def get_bin_edges(gradient_images: list, num_bins=200):
        """Compute bin edges from multiple gradient sets."""
        flattened = np.concatenate([img.flatten() for img in gradient_images])
        _, bin_edges = np.histogram(flattened, bins=num_bins)
        return bin_edges

    @staticmethod
    def border_free(imgs: np.ndarray, border_size: int):
        """Remove borders from the images."""
        return imgs[:, border_size:-border_size, border_size:-border_size, :]

    @staticmethod
    def wiener_entropy(hist: np.ndarray, eps=1e-12):
        """Compute Wiener entropy for the histogram."""
        w = np.hanning(len(hist))
        X = np.fft.rfft(hist * w)
        P = np.abs(X) ** 2 + eps
        geom_mean = np.exp(np.mean(np.log(P)))
        arith_mean = np.mean(P)
        return 1.0 - float(geom_mean / (arith_mean + eps))

    def _gradients_along_tile_grid(self, offset: int, channels=None):
        """
        Sample gradients along tile grid with optional channels.

        channels: int, list/tuple of ints, or None (all channels)
        """
        if channels is None:
            channels = list(range(self.gradients_x.shape[-1]))
        elif isinstance(channels, int):
            channels = [channels]

        grad_x_slice = self.gradients_x[:, :, offset::self.tile_size, channels]
        grad_y_slice = self.gradients_y[:, offset::self.tile_size, :, channels]

        return np.concatenate([grad_x_slice.flatten(), grad_y_slice.flatten()])


    def get_gradients_at(self, position="edge", channels=None):
        """
        Get gradients sampled at specific tile positions.

        position: "edge", "middle", or int (tile offset)
        channels: int, list/tuple of ints, or None (all channels)
        """
        if isinstance(position, str):
            position = position.lower()
            if position == "edge":
                offset = self.tile_size - 1
            elif position == "middle":
                offset = self.tile_size // 2 - 1
            else:
                raise ValueError("position must be 'edge', 'middle', or an integer")
        elif isinstance(position, int):
            offset = position
        else:
            raise TypeError("position must be a string or integer")

        return self._gradients_along_tile_grid(offset, channels=channels)



    def get_peakiness_scores(self, eps=1e-12):
        """Compute peakiness scores using Wiener entropy."""
        scores = []
        for x in [self.histogram_edges,
                  self.histogram_middle,
                  self.histogram_middle - self.histogram_edges]:
            scores.append(GradientUtils.wiener_entropy(x, eps=eps))
        return scores


