import torch
from torch import nn
import numpy as np
from .config import th_float
class voigt_expand(nn.Module):
    def __init__(self,
                 ndim = 5040,
                 nmin = 0,
                 nmax=90,
                 gauss_shape = 0.001,
                 loren_shape = 1.25,
                 voigt_par = 0.5):
        
        super().__init__()
        self.x = np.linspace(nmin, nmax, ndim)
        self.gauss_shape = gauss_shape
        self.loren_shape = loren_shape
        self.voigt_par = voigt_par
        
    def gaussian_expand(self, xc, yc):
        #z = np.exp( - np.power((self.x[:, np.newaxis] - xc), 2 )) / np.sqrt(self.gauss_shape / yc) @ yc[:, np.newaxis]
        z = np.exp( - np.power((self.x[:, np.newaxis] - xc), 2) / np.sqrt( self.gauss_shape / yc) )  @ yc[:, np.newaxis]
        return z / z.max() *100
        
    def lorentzian_expand(self, xc, yc):
        #z =  (1 / (np.power((self.x[:, np.newaxis] - xc), 2) + self.loren_shape * 4 / np.power(yc, 2)[np.newaxis, :])) @ (self.loren_shape / yc[:, np.newaxis] / (2 * np.pi))
        #z =  (1 / (np.power((self.x[:, np.newaxis] - xc), 2) +  self.loren_shape / np.power(yc, 2)[np.newaxis, :])) @ (1 / yc[:, np.newaxis] / (2 * np.pi))
        z =  (yc[np.newaxis, :] / (np.power((self.x[:, np.newaxis] - xc), 2) +  self.loren_shape / np.power(yc, 2)[np.newaxis, :])) @ (1 / yc[:, np.newaxis] / (2 * np.pi))
        return z / z.max() *100
        
    def forward(self, xc, yc):
        x = self.voigt_par * self.gaussian_expand(xc, yc) + (1 - self.voigt_par) * self.lorentzian_expand(xc, yc)
        return torch.tensor(x, dtype = torch.float32).squeeze()


#input x: np.array, y: np.array
#need high resolution to present exactly relativity intensity.
class simple_profile():
    def __init__(
                self,
                nmax=180,    #range
                nmin=0,      #range
                ndim=5040,   #resolution
                fwhm=0.2,
                voigt_par=0.5,
        ):
        self.x = np.linspace(nmin, nmax, ndim)
        self.fwhm = fwhm
        self.voigt_par = voigt_par
        
    def gaussian(self, xc, yc):
        z = np.exp( -4 * np.log(2) * np.power((self.x[:, np.newaxis] - xc) / self.fwhm, 2)) @ yc[:, np.newaxis]
        return z
        
    def lorentzian(self, xc, yc):
        z = (1 / (1 + 4 *(np.power((self.x[:, np.newaxis] - xc) / self.fwhm, 2)))) @ yc[:, np.newaxis]
        return z
        
    def pseudo_voigt(self, xc, yc):
        z = (1. - self.voigt_par) * self.gaussian(xc, yc) + self.voigt_par * self.lorentzian(xc, yc)
        return z