import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import affine
import torch.fft as fft
import random


class Tracker:
    def __init__(self, frame, top, left, height, width, p, nu, sigma):
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.p = p
        self.nu = nu
        self.sigma = sigma
        self.k = 0.04
        self.A = torch.zeros(3, 3, height, width, dtype=torch.complex64)
        self.B = torch.zeros(3, 3, height, width, dtype=torch.complex64)
        self.A0 = torch.zeros(3, 3, height, width, dtype=torch.complex64)
        self.B0 = torch.zeros(3, 3, height, width, dtype=torch.complex64)
        Gc = self._make_gaussian(self.height, self.width, self.sigma)
        Gc = (Gc - Gc.min()) / (Gc.max() - Gc.min())
        self.Gc_ = fft.fft2(Gc)
        self.mask = self._make_mask(self.height, self.width)
        frame_pt = transforms.ToTensor()(frame)
        F = self._get_box(frame_pt)
        self._train(F, 1.0)
        self.A0 = self.A.clone().detach()
        self.B0 = self.B.clone().detach()

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
        idx_flat = torch.argmax(tensor).item() % (tensor.shape[1] * tensor.shape[2])
        shift_y = idx_flat // tensor.shape[2] - tensor.shape[1] // 2
        shift_x = idx_flat % tensor.shape[2] - tensor.shape[2] // 2
        return shift_y, shift_x

    def _get_box(self, frame_pt):
        return frame_pt[:, self.top:self.top + self.height, self.left:self.left + self.width]

    def _preprocess_box(self, F):
        std, mean = torch.std_mean(F, dim=(-2, -1))
        return ((F - mean[:, None, None]) / (std[:, None, None] + 1E-8)) * self.mask

    def _generate_params(self):
        angles = random.choices([5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60], k=self.p)
        blurs = random.choices([(3, 0.2), (3, 1), (5, 2), (7, 3)], k=self.p)
        shifts = random.choices([[6, 6], [-6, 6], [6, -6], [-6, -6]], k=self.p)
        return angles, blurs, shifts

    def _train(self, F, nu):
        _, height, width = F.shape
        A_new = torch.zeros(3, 3, height, width, dtype=torch.complex64)
        B_new = torch.zeros(3, 3, height, width, dtype=torch.complex64)
        angles, blurs, shifts = self._generate_params()
        scales = [1.0 - self.k,
                  1.0,
                  1.0 + self.k]
        for i in range(self.p):
            for j in range(3):
                Fp = affine(F,
                            angle=angles[i],
                            translate=[0, 0],
                            scale=scales[j],
                            shear=shifts[i])
                Fp = transforms.GaussianBlur(
                    kernel_size=blurs[i][0],
                    sigma=blurs[i][1])(Fp)
                Fp = self._preprocess_box(Fp)
                Fp_ = fft.fft2(Fp)
                A_new[j] += self.Gc_ * torch.conj(Fp_)
                B_new[j] += Fp_ * torch.conj(Fp_)
        self.A = nu * A_new + (1 - nu) / 2 * (self.A + self.A0)
        self.B = nu * B_new + (1 - nu) / 2 * (self.B + self.B0)

    def _update_box(self, frame_pt):
        F = self._get_box(frame_pt)
        F = self._preprocess_box(F)
        F_ = fft.fft2(F)
        W_ = self.A / (self.B + 1E-8)
        G_ = W_ * F_
        G = torch.sum(fft.ifft2(G_).real, dim=1)
        shift_y, shift_x = self._get_shift(G)
        self.top += shift_y
        self.left += shift_x

    def next(self, frame):
        frame_pt = transforms.ToTensor()(frame)
        self._update_box(frame_pt)
        F = self._get_box(frame_pt)
        self._train(F, self.nu)
        return self.top, self.left
