import torch
import torchvision.transforms as transforms
import torch.fft as fft


class Tracker:
    def __init__(self, frame, box_top, box_left, p, nu):
        self.box_top = box_top
        self.box_left = box_left
        self.p = p
        self.nu = nu
        self.A = 0
        self.B = 0
        self.mask = self._make_mask()
        self.Gc = self._make_gaussian()
        self.Gc_ = fft.fft2(self.Gc)

        F = frame[self.box_top:self.box_top + 64, self.box_left:self.box_left + 64]
        self._train(F, 1.0)

    @staticmethod
    def _make_mask():
        window_1d = torch.hamming_window(64)
        window_2d = torch.outer(window_1d, window_1d)
        return window_2d

    @staticmethod
    def _make_gaussian():
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, steps=64),
            torch.linspace(-1, 1, steps=64),
            indexing="ij"
        )
        sigma = 0.7
        gaussian = torch.exp(- (x ** 2 + y ** 2) / sigma ** 2)
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
            Fp = transforms.RandomAffine(degrees=(-30, 30), scale=(0.8, 1.2), shear=(-10, 10))(F.unsqueeze(0))[0]
            Fp_ = fft.fft2(Fp * self.mask)
            Anew += self.Gc_ * torch.conj(Fp_)
            Bnew += Fp_ * torch.conj(Fp_)

        self.A = nu * Anew + (1 - nu) * self.A
        self.B = nu * Anew + (1 - nu) * self.B

    def next(self, frame):
        F = frame[self.box_top:self.box_top + 64, self.box_left:self.box_left + 64]
        F_ = fft.fft2(F * self.mask)
        self.B += 0.0000001
        G_ = self.A / self.B * F_
        G = fft.ifft2(G_).real
        row, col = self._argmax(G)
        self.box_top = self.box_top + row - 32
        self.box_left = self.box_left + col - 32
        self._train(F, self.nu)
        return self.box_top, self.box_left
