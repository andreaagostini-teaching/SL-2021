import aa_soundfile as sf
import numpy as np
import scipy as sp
import scipy.interpolate as spi
import math
import matplotlib.pyplot as plt

SR = 44100


def phasorS(freq, dur):
    return (np.arange(dur * SR) / SR * freq) % 1


def phasorM(freq, ph = 0):
    """
    freq is an ndarray of frequencies
    """
    p = np.zeros(freq.shape)
    n = 0
    for f in freq:
        p[n] = ph
        ph += f / SR
        ph %= 1
        n += 1
    return p


def sine(ph):
    """
    ph is a phasor
    """
    a = np.sin(ph * math.tau)
    return a


def bpf(bp, stretch = 1):
    """
    bp has the form
    [ start, deltaT, value, ... ]
    (see csound's linseg)
    """
    l = len(bp)
    a = np.array([])
    c = bp[0]
    n = 1
    while n < l - 1:
        t = bp[n]
        d = bp[n + 1]
        a = np.hstack((a, np.linspace(c, d, t * stretch * SR)))
        c = d
        n += 2
    return a


def offset(a, t):
    return np.hstack((np.zeros(t*SR), a))


def mix(a):
    """
    a has the form
    [ [ snd, time ], [snd, time], ... ]
    or
    [ snd, snd, [snd, time] ]
    if time is not given it is assumed to be 0
    """

    r = np.array([])

    for s in a:
        try:
            print('s: ', s)
            iter(s[0]) # is s[0] iterable --- that is, not a float?
            snd = s[0]
            t = 0 if len(s) == 1 else int(s[1] * SR)
            nl = t + len(snd)
            if nl > len(r): r.resize(nl)
            r[t:nl] += snd
        except TypeError: # it is not an iterable
            print(f'  except: s = {s}')
            nl = int(len(s))
            print(f'  nl = {nl}, len(r) = {len(r)}')
            if nl > len(r): r.resize(nl)
            r[0:nl] += s

    return r


def resample(a, ratio=1, kind='cubic'):
    """
    ratio can be thought of as a transposition factor (see groove~)
    The final duration will be len(a) / ratio
    """

    x = np.arange(len(a))
    f = spi.interp1d(x, a, kind=kind)
    x2 = np.linspace(0, len(a) - 1, int(len(a) / ratio + 0.5))
    return f(x2)


def delayS(x, time = 0, fb = 0, dur = -1, func = None, funcData = None):
    """
    dur is the duration of the resulting sound
     if < 0 it will be set to the duration of a + time
    """
    time = int(time * SR)
    if dur < 0:
        dur = len(a) * SR + time
    else:
        dur *= SR
    y = np.zeros(dur)
    if func == None:
        for wp in range(time, len(x)):
            rp = wp-time
            y[wp] = x[rp] + y[rp] * fb
        for wp in range(len(x), dur):
            rp = wp-time
            y[wp] = y[rp] * fb
    else:
        for n in range(time, dur):
            rp = n-time
            y[n] = x[rp] + func(y[rp], funcData)
    return y


def delayM(a, time, fb, dur, func = None, funcData = None):
    d = np.zeros(dur)
    t = 0
    if func == None:
        for n in range(dur):
            if n < len(time): t = time[n] * SR
            if t >= 0 and t <= n:
                rp = n - t
                d[n] = a[rp] + d[rp] * fb
            else:
                d[n] = 0
    else:
        for n in range(dur):
            if n < len(time): t = time[n] * SR
            if t >= 0 and t <= n:
                rp = n - t
                d[n] = a[rp] + func(d[rp], funcData)
            else:
                d[n] = 0
    return d


def biquad(a, k=(1, 0, 0, 0, 0), mem=[0, 0, 0, 0]):
    """
    y[n] = a0 * x[n] + a1 * x[n-1] + a2 * x[n-2] - b1 * y[n-1] - b2 * y[n-2]
    coeff is (a0, a1, a2, b1, b2)
    mem is [x[n-1], x[n-2], y[n-1], y[n-2]]
    """
    try:
        iter(a)
    except TypeError:
        a = np.array([a])

    y = np.zeros(a.shape)

    print(a.shape)
    print(a)
    print(y)

    n = 0
    for x in a:
        y[n] = x * k[0] + mem[0] * k[1] + mem[1] * k[2] - mem[2] * k[3] - mem[3] * k[4]
        mem = [ x, mem[0], y[n], mem[2] ]
        n += 1

    return y


def scale(a, in1 = 0, in2 = 1, out1 = 0, out2 = 1):
    """
    parameters can be scalars or arrays
    """
    return (a - in1) / (in2 - in1) * (out2 - out1) + out1


def noise(dur, lo = -1, hi = 1):
    """
    Generates a matrix of white noise of <dur> seconds
    (if needed, I can set a range for the generation)
    ex:
    n = au.noise(10) # 10 seconds of noise
    """
    dur *= SR
    a = scale(np.random.rand(int(dur)), 0, 1, lo, hi)


def normalize(a, lo = -1, hi = 1):
    lowest = abs(a.min())
    highest = abs(a.max())
    m = max(lowest, highest)
    return scale(a, -m, m, lo, hi)


def reson(f = 1000, Q = 1):
    """
    calculates a tuple of coefficients for biquad
    --- shamelessly stolen from the gen~ filter examples
    """
    omega = f * math.tau / SR;
    sn = math.sin(omega);
    cs = math.cos(omega);
    alpha = sn * 0.5 / Q;

    b0 = 1./(1. + alpha);
    a0 = alpha * Q * b0;
    a1 = 0.;
    a2 = -alpha * Q * b0;
    b1 = -2. * cs * b0;
    b2 = (1. - alpha) * b0;

    return (a0, a1, a2, b1, b2)


def lowpass(f = 1000, Q = 1):
    """
    calculates a tuple of coefficients for biquad
    --- shamelessly stolen from the gen~ filter examples
    """
    omega = cf * math.tau / SR;
    sn = math.sin(omega);
    cs = math.cos(omega);
    igain = 1.0/gain;
    one_over_Q = 1./Q;
    alpha = sn * 0.5 * one_over_Q;

    b0 = 1./(1. + alpha);
    a2 = ((1 - cs) * 0.5) * b0;
    a0 = a2;
    a1 = (1. - cs) * b0;
    b1 = (-2. * cs) * b0;
    b2 = (1. - alpha) * b0;

    return (a0, a1, a2, b1, b2)


def show(a):
    """
    displays a quick and dirty graph of a matrix
    """
    fig, ax = plt.subplots()  # Create a figure containing a single axis
    x = np.linspace(0, len(a) / SR, len(a))
    ax.plot(x, a)  # Plot some data on the axes.
    plt.show()
