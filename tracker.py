import torch
import torchvision.transforms as transforms
import torch.fft as fft


class Tracker:
    def __init__(self, frame, top, left, height, width, p, nu, sigma):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.p = p
        self.nu = nu
        self.sigma = sigma
        self.A = 0
        self.B = 0
        Gc = self._make_gaussian(self.height, self.width, self.sigma)
        Gc = (Gc - Gc.min()) / (Gc.max() - Gc.min())
        self.Gc_ = fft.fft2(Gc)
        self.mask = self._make_mask(self.height, self.width)
        frame_pt = transforms.Grayscale()(transforms.ToTensor()(frame)).squeeze(0)
        F = self._get_box(frame_pt)
        self._train(F, 1.0)

    @staticmethod
    def _make_mask(height, width):
        window_1d_y = torch.hamming_window(height)
        window_1d_x = torch.hamming_window(width)
        window_2d = torch.outer(window_1d_y, window_1d_x)
        return window_2d

    @staticmethod
    def _make_gaussian(height, width, sigma):
        y, x = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij"
        )
        center_y = height // 2
        center_x = width // 2
        gaussian = torch.exp(- ((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
        return gaussian

    @staticmethod
    def _get_shift(tensor):
        idx_flat = torch.argmax(tensor)
        shift_y = idx_flat.item() // tensor.shape[1] - tensor.shape[0] // 2
        shift_x = idx_flat.item() % tensor.shape[1] - tensor.shape[1] // 2
        return shift_y, shift_x

    def _get_box(self, frame_pt):
        return frame_pt[self.top:self.top + self.height, self.left:self.left + self.width]

    def _preprocess_box(self, F):
        std, mean = torch.std_mean(F)
        return ((F - mean) / (std + 1E-8)) * self.mask

    def _train(self, F, nu):
        A_new = 0
        B_new = 0
        for _ in range(self.p):
            Fp = transforms.RandomAffine(degrees=(-180 / 16, 180 / 16), shear=(-10, 10))(F.unsqueeze(0)).squeeze(0)
            Fp = self._preprocess_box(Fp)
            Fp_ = fft.fft2(Fp)
            A_new += self.Gc_ * torch.conj(Fp_)
            B_new += Fp_ * torch.conj(Fp_)
        self.A = nu * A_new + (1 - nu) * self.A
        self.B = nu * B_new + (1 - nu) * self.B

    def _update_box(self, frame_pt):
        F = self._get_box(frame_pt)
        F = self._preprocess_box(F)
        F_ = fft.fft2(F)
        W_ = self.A / (self.B + 1E-8)
        G_ = W_ * F_
        G = fft.ifft2(G_).real
        shift_y, shift_x = self._get_shift(G)
        self.top += shift_y
        self.left += shift_x

    def next(self, frame):
        frame_pt = transforms.Grayscale()(transforms.ToTensor()(frame)).squeeze(0)
        self._update_box(frame_pt)
        F = self._get_box(frame_pt)
        self._train(F, self.nu)
        return self.top, self.left
