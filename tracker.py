import torch
import torchvision.transforms as transforms
import torch.fft as fft


class Tracker:
    def __init__(self, frame, box_top, box_left, height, width, p, nu):
        self.box_top = box_top
        self.box_left = box_left
        self.height = height
        self.width = width
        self.p = p
        self.nu = nu
        self.A = 0
        self.B = 0
        self.mask = self._make_mask(self.height, self.width)
        self.Gc = self._make_gaussian(self.height, self.width)
        self.Gc_ = fft.fft2(self.Gc)

        F = frame[self.box_top:self.box_top + self.height, self.box_left:self.box_left + self.width]
        self._train(F, 1.0)

    @staticmethod
    def _make_mask(height, width):
        window_1d_y = torch.hamming_window(height)
        window_1d_x = torch.hamming_window(width)
        window_2d = torch.outer(window_1d_y, window_1d_x)
        return window_2d

    @staticmethod
    def _make_gaussian(height, width):
        y, x = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij"
        )
        center_y = height // 2
        center_x = width // 2
        sigma = max(height, width) / 4
        gaussian = torch.exp(- ((x - center_x) ** 2 + (y - center_y) ** 2) / sigma ** 2)
        return gaussian

    @staticmethod
    def _argmax(tensor):
        idx_flat = torch.argmax(tensor)
        row = idx_flat.item() // tensor.shape[1]
        col = idx_flat.item() % tensor.shape[1]
        return row, col

    def _train(self, F, nu):
        F_ = fft.fft2(F * self.mask)
        Anew = self.Gc_ * torch.conj(F_)
        Bnew = F_ * torch.conj(F_)

        for _ in range(self.p):
            Fp = transforms.RandomAffine(
                degrees=(-30, 30),
                scale=(0.8, 1.2),
                shear=(-10, 10)
            )(F.unsqueeze(0))[0]
            Fp_ = fft.fft2(Fp * self.mask)
            Anew += self.Gc_ * torch.conj(Fp_)
            Bnew += Fp_ * torch.conj(Fp_)

        self.A = nu * Anew + (1 - nu) * self.A
        self.B = nu * Anew + (1 - nu) * self.B

    def next(self, frame):
        F = frame[self.box_top:self.box_top + self.height, self.box_left:self.box_left + self.width]
        F_ = fft.fft2(F * self.mask)
        self.B += 0.0000001
        G_ = self.A / self.B * F_
        G = fft.ifft2(G_).real
        row, col = self._argmax(G)
        self.box_top = self.box_top + row - self.height // 2
        self.box_left = self.box_left + col - self.width // 2
        self._train(F, self.nu)
        return self.box_top, self.box_left
