import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import affine
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
        self.A = torch.zeros(3, height, width, dtype=torch.complex64)
        self.B = torch.zeros(3, height, width, dtype=torch.complex64)
        Gc = self._make_gaussian(self.height, self.width, self.sigma)
        Gc = (Gc - Gc.min()) / (Gc.max() - Gc.min())
        self.Gc_ = fft.fft2(Gc)
        self.mask = self._make_mask(self.height, self.width)
        frame_pt = transforms.ToTensor()(frame)
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
        return frame_pt[:, self.top:self.top + self.height, self.left:self.left + self.width]

    def _preprocess_box(self, F):
        std, mean = torch.std_mean(F, dim=(-2, -1))
        return ((F - mean[:, None, None]) / (std[:, None, None] + 1E-8)) * self.mask

    def _train(self, F, nu):
        _, height, width = F.shape
        A_new = torch.zeros(3, height, width, dtype=torch.complex64)
        B_new = torch.zeros(3, height, width, dtype=torch.complex64)
        angles = [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60]
        blurs = [(3, 0.2), (3, 1), (5, 2), (7, 3)]
        shift = [[6, 6], [-6, 6], [6, -6], [-6, -6]]
        angles_idx = torch.randint(len(angles), (self.p,))
        blurs_idx = torch.randint(len(blurs), (self.p,))
        shift_idx = torch.randint(len(shift), (self.p,))
        scales = torch.FloatTensor(self.p).uniform_(0.8, 1.2)
        for i in range(self.p):
            Fp = affine(F,
                        angle=angles[angles_idx[i]],
                        translate=[0, 0],
                        scale=scales[i],
                        shear=shift[shift_idx[i]])
            Fp = transforms.GaussianBlur(
                kernel_size=blurs[blurs_idx[i]][0],
                sigma=blurs[blurs_idx[i]][1])(Fp)
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
        G = torch.sum(fft.ifft2(G_).real, dim=0)
        shift_y, shift_x = self._get_shift(G)
        self.top += shift_y
        self.left += shift_x

    def next(self, frame):
        frame_pt = transforms.ToTensor()(frame)
        self._update_box(frame_pt)
        F = self._get_box(frame_pt)
        self._train(F, self.nu)
        return self.top, self.left
