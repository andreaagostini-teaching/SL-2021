import aa_soundfile as sf
import numpy as np
import scipy as sp
import scipy.interpolate as spi
import math
import matplotlib.pyplot as plt

class Audionode(object):
    def __init__(self, sr = 44100):
        self.sr = 44100

class Phasor(Audionode):
    def __init__(self, freq = 0, phase = 0, sr = 44100):
        super(Phasor, self).__init__(sr)
        self.freq = freq
        self.phase = phase

    def process(self, freq = None, dur = 0):
        if type(freq) == type(None):
            durSamps = dur * self.sr
            incr = self.sr / self.freq
            out = (np.arange(dur + 1) / incr + self.phase) % 1
            self.phase = out[-1]
            out.resize(out.shape[0] - 1)
        else:
            out = np.zeros(freq.shape)
            n = 0
            ph = self.phase
            for f in freq:
                out[n] = ph
                ph = (ph + f / self.sr) % 1
                n += 1
            self.phase = ph
        return out


class Sine(Audionode):
    def __init__(self, sr = 44100):
        super(Sine, self).__init__(sr)

    def process(self, ph):
        out = np.sin(ph * math.tau)
        return out


class Bpf(Audionode):
    def __init__(self, bp, stretch = 1, sr = 44100):
        super(Bpf, self).__init__(sr)
        self.bp = bp
        self.stretch = stretch

    def process(self):
        l = len(self.bp)
        out = np.array([])
        c = self.bp[0]
        n = 1
        while n < l - 1:
            t = self.bp[n]
            d = self.bp[n + 1]
            out = np.hstack((out, np.linspace(c, d, t * self.stretch * self.sr)))
            c = d
            n += 2
        return out
