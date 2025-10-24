import numpy as np

class GradientUtils2D:
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

        # remove border artefacts
        self.imgs_without_borders = GradientUtils2D.border_free(self.imgs, self.border_size)

        # gradients
        self.gradients_x, self.gradients_y = GradientUtils2D.compute_gradients(self.imgs, self.border_size)

        # gradients along tile grid
        self.gradients_edges = self._gradients_along_tile_grid(offset=self.tile_size - 1)
        self.gradients_middle = self._gradients_along_tile_grid(offset=self.tile_size // 2 - 1)

        # compute bin edges (if not given)
        self._bin_edges = bin_edges
        if self._bin_edges is None:
            self._bin_edges = GradientUtils2D.get_bin_edges(
                [self.gradients_x, self.gradients_y, self.gradients_edges, self.gradients_middle],
                num_bins=200
            )
        
        # histograms
        self.histogram_edges = GradientUtils2D.compute_histograms(self.gradients_edges, self._bin_edges)
        self.histogram_middle = GradientUtils2D.compute_histograms(self.gradients_middle, self._bin_edges)
    
    # FIXED: Added method to create bin edges (was missing!)
    def make_bin_edges(self, n_bins=2000):
        """Create bin edges from the gradient data."""
        return GradientUtils2D.get_bin_edges(
            [self.gradients_x, self.gradients_y, self.gradients_edges, self.gradients_middle],
            num_bins=n_bins
        )
        
    @staticmethod
    def compute_histograms(gradients: np.ndarray, bin_edges: np.ndarray):
        """
        Compute histograms for gradients.
        Args:
            gradients (numpy array): gradients
            bin_edges (numpy array): edges of the histogram bins.
        Returns:
            histograms (tuple): histograms for edges and middle gradients.
        """
        return np.histogram(gradients, bins=bin_edges)[0]

    @staticmethod
    def compute_gradients(imgs: np.ndarray, border_size: int = 0):
        """Compute horizontal and vertical gradients for an image batch."""
        wb = GradientUtils2D.border_free(imgs, border_size)
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
        """Compute Wiener entropy for the histogram.(not flattened array)"""
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
        Get collection of absolute gradients sampled at specific tile positions.

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



    def get_peakiness_scores(self, histogram_edges, histogram_middle, eps=1e-12):
        """Compute peakiness scores using Wiener entropy."""
        scores = []
        for x in [histogram_edges,
                  histogram_middle,
                  histogram_middle - histogram_edges]:
            scores.append(GradientUtils2D.wiener_entropy(x, eps=eps))
        return scores
    
import numpy as np

class GradientUtils3D:
    """
    Class to compute 3D tiling statistics such as gradient histograms and peakiness scores
    for volumetric images (shape: [B, Z, Y, X, C]).
    """

    def __init__(self, imgs: np.ndarray, tile_size=(9, 32, 32), border_size=None, bin_edges=None):
        """
        Initialize GradientUtils3D.

        Args:
            imgs (np.ndarray): 5D image array [B, Z, Y, X, C].
            tile_size (tuple[int]): tile size along (Z, Y, X). e.g. (9, 32, 32)
            border_size (tuple[int] or None): border width to ignore per axis.
                                              Defaults to half of each tile_size.
            bin_edges (np.ndarray or None): histogram bin edges.
        """
        self.imgs = imgs
        self.tile_size = np.array(tile_size)
        if border_size is None:
            self.border_size = self.tile_size // 2
        else:
            self.border_size = np.array(border_size)

        # remove borders
        self.imgs_wo_borders = GradientUtils3D.border_free(imgs, self.border_size)

        # compute gradients along 3D axes
        self.grad_z, self.grad_y, self.grad_x = GradientUtils3D.compute_gradients(imgs, self.border_size)

        # gradients along tile grid
        self.grad_edges = self._gradients_along_tile_grid(offset=self.tile_size - 1)
        self.grad_middle = self._gradients_along_tile_grid(offset=self.tile_size // 2 - 1)

        # compute bin edges if not provided
        self._bin_edges = bin_edges
        if self._bin_edges is None:
            self._bin_edges = GradientUtils3D.get_bin_edges(
                [self.grad_x, self.grad_y, self.grad_z, self.grad_edges, self.grad_middle],
                num_bins=200
            )

        # histograms
        self.histogram_edges = GradientUtils3D.compute_histograms(self.grad_edges, self._bin_edges)
        self.histogram_middle = GradientUtils3D.compute_histograms(self.grad_middle, self._bin_edges)

    # ---------- STATIC METHODS ----------

    @staticmethod
    def border_free(imgs: np.ndarray, border_size):
        """Remove borders from a 3D image [B, Z, Y, X, C]."""
        bz, by, bx = border_size
        return imgs[:, bz:-bz, by:-by, bx:-bx, :]

    @staticmethod
    def compute_gradients(imgs: np.ndarray, border_size):
        """Compute 3D gradients along Z, Y, X for batch of volumes."""
        wb = GradientUtils3D.border_free(imgs, border_size)
        grad_z = wb[:, 1:, :, :, :] - wb[:, :-1, :, :, :]
        grad_y = wb[:, :, 1:, :, :] - wb[:, :, :-1, :, :]
        grad_x = wb[:, :, :, 1:, :] - wb[:, :, :, :-1, :]
        return grad_z, grad_y, grad_x

    @staticmethod
    def get_bin_edges(gradient_volumes: list, num_bins=200):
        """Compute bin edges from multiple 3D gradient arrays."""
        flattened = np.concatenate([v.flatten() for v in gradient_volumes])
        _, bin_edges = np.histogram(flattened, bins=num_bins)
        return bin_edges

    @staticmethod
    def compute_histograms(gradients: np.ndarray, bin_edges: np.ndarray):
        """Compute histogram for 3D gradient data."""
        return np.histogram(gradients, bins=bin_edges)[0]

    @staticmethod
    def wiener_entropy(hist: np.ndarray, eps=1e-12):
        """Compute Wiener entropy for histogram."""
        w = np.hanning(len(hist))
        X = np.fft.rfft(hist * w)
        P = np.abs(X) ** 2 + eps
        geom_mean = np.exp(np.mean(np.log(P)))
        arith_mean = np.mean(P)
        return 1.0 - float(geom_mean / (arith_mean + eps))

    # ---------- INTERNAL METHODS ----------

    def _gradients_along_tile_grid(self, offset, channels=None):
        """
        Sample gradients along 3D tile grid.
        offset: array-like of (z_offset, y_offset, x_offset)
        """
        oz, oy, ox = offset
        if channels is None:
            channels = list(range(self.grad_x.shape[-1]))
        elif isinstance(channels, int):
            channels = [channels]

        grad_x_slice = self.grad_x[:, :, :, ox::self.tile_size[2], channels]
        grad_y_slice = self.grad_y[:, :, oy::self.tile_size[1], :, channels]
        grad_z_slice = self.grad_z[:, oz::self.tile_size[0], :, :, channels]

        return np.concatenate([
            grad_x_slice.flatten(),
            grad_y_slice.flatten(),
            grad_z_slice.flatten()
        ])

    # ---------- PUBLIC API ----------

    def make_bin_edges(self, n_bins=2000):
        """Create bin edges from gradient data."""
        return GradientUtils3D.get_bin_edges(
            [self.grad_x, self.grad_y, self.grad_z, self.grad_edges, self.grad_middle],
            num_bins=n_bins
        )

    def get_peakiness_scores(self, histogram_edges, histogram_middle, eps=1e-12):
        """Compute peakiness scores using Wiener entropy."""
        scores = []
        for x in [histogram_edges,
                  histogram_middle,
                  histogram_middle - histogram_edges]:
            scores.append(GradientUtils3D.wiener_entropy(x, eps=eps))
        return scores

    def get_gradients_at(self, position="edge", channels=None):
        """
        Get gradients sampled at specific tile positions.
        position: 'edge', 'middle', int, or tuple/list of (z,y,x) offsets
        """
        if isinstance(position, str):
            position = position.lower()
            if position == "edge":
                offset = self.tile_size - 1
            elif position == "middle":
                offset = self.tile_size // 2 - 1
            else:
                raise ValueError("position must be 'edge', 'middle', int, or tuple of ints")
        elif isinstance(position, int):
            # Same offset in all dimensions
            offset = np.array([position, position, position])
        elif isinstance(position, (list, tuple, np.ndarray)):
            offset = np.array(position)
        else:
            raise TypeError("position must be string, int, or tuple/list of ints")

        return self._gradients_along_tile_grid(offset, channels=channels)



