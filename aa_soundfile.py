"""
aa_soundfile: a lightweight module for reading and writing sound files

 Dependencies:
  numpy
"""

import aifc
import wave
import sys
import os
import numpy as np


def read(fileName, dtype='float32'):
    """
    Reads a sound file in wav or aiff format, and returns it as a normalized numpy array.
    Also returns a dict with a few of the file's metadata.

    Returns:
     . A numpy.ndarray of the specified dtype, with one sub-array per channel:
       [ [ smp1ch1, smp2ch1, ... ], [ smp1ch2, smp2ch2, ...], ... ]
     . A dictionary with the following entries:
       'sr': the file's sample rate
       'nframes': the number of audio frames in the file
            (that is, the length of the 1st axis of the array)
       'dur': the duration of the file in seconds
       'chans': the number of channels in the file
            (that is, the length of the 2nd axis of the array)
       'res': the resolution in bytes of the file (1, 2 or 3)

    Example:
    snd, meta = aa_soundfile.read('chopin')
    """

    ext = os.path.splitext(fileName)[1]

    if ext == '.wav':
        f = wave.open(fileName, 'rb')
        order = '<'
    elif ext == '.aif' or ext == '.aiff':
        f = aifc.open(fileName, 'rb')
        order = '>'
    else: return None, None

    meta = dict()
    sr = meta['sr'] = f.getframerate()
    frames = meta['nframes'] = f.getnframes()
    meta['dur'] = frames / sr
    chans = meta['chans'] = f.getnchannels()
    res = meta['res'] = f.getsampwidth()

    #print("durata: %s frame, %s secondi" % (frames, sec))
    #print("canali: %s" % chans)
    #print("risoluzione: %s bit" % (res * 8))
    fr = f.readframes(frames)


    if res == 3:
        dt = np.dtype('uint8')
        dt = dt.newbyteorder(order)
        raw = np.frombuffer(fr, dtype=dt)
        raw = raw.reshape((frames, chans * res))
        raw = raw.T
        raw = raw.reshape((chans, res, frames))
        raw = np.array(raw, dtype='int32')
        for ch in raw:
            ch[0] = np.where(ch[0] < 128, ch[0], ch[0] - 256)
        cooked = np.array([x[0] * 65536 + x[1] * 256 + x[2] for x in raw])
        norm = np.array(cooked / (2 ** 23), dtype=dtype)

    else:
        if res == 1: dt = np.dtype('int8')
        elif res == 2: dt = np.dtype('int16')
        dt = dt.newbyteorder(order)
        f.close()

        raw = np.frombuffer(fr, dtype=dt)
        norm = np.array(raw / (2 ** (res * 8 - 1)), dtype=dtype)
        norm = norm.reshape((frames, chans)).T

    return norm, meta




def flat(snd):
    """
    Returns a flat ndarray containing the average of the individual channels of snd.
    This is the format typically required by librosa and essentia.

    Arguments:
     . snd: an ndarray as returned by read()

    Returns:
     . A flat ndarray:
       [ smp1, smp2, ... ]
    """

    return snd.mean(axis=0)




def mono(snd):
    """
    Returns a mono version of snd

    Arguments:
     . snd: a ndarray as returned by read()

    Returns:
     . A mono version of snd:
       [ [ smp1, smp2, ... ] ]
    """

    return flat(snd).reshape(1, snd[0].size)




def write(snd, fileName, sr = 44100):
    """
    Writes a 16-bit soundfile in wav or aiff format.

    Arguments:
     . snd: a ndarray as returned by read(), or a flat ndarray as returned by flat()
     . filename: the file name
     . sr: the sample rate
    """

    ext = os.path.splitext(fileName)[1]

    if ext == '.wav': f = wave.open(fileName, 'wb')
    elif ext == '.aif' or ext == '.aiff': f = aifc.open(fileName, 'wb')
    else: return None, None

    if len(snd.shape) == 1:
        snd = snd.reshape(1, snd.size)

    frames = snd.shape[1]
    f.setnchannels(snd.shape[0])
    f.setnframes(frames)
    f.setframerate(sr)
    f.setsampwidth(2)

    cooked = np.array(snd.T * 32767, dtype='int16')
    if ext != '.wav': cooked.byteswap(True)
    f.writeframes(cooked)
