import numpy as np

from progress.bar import Bar
from matplotlib import pyplot as plt
from numpy import linspace, cos, sin, pi, sqrt
from scipy.signal import remez, lfilter

from spec_plot import spec_plot
from bandpass_limiter import bandpass_limiter


def demodulate_fm(fm_mod, fs, BW = 200000,
                  debug = False, deemph = True, fc = 0,
                  BPL = True, progress = False):
  '''
  '''
  if progress:
    bar = Bar('Demodulating FM...', max = 5)
    bar.next()

  fs = float(fs)
  BW = float(BW)

  if BPL == True:
    fm_mod = bandpass_limiter(fm_mod, BW, fc, fs, debug)
  if progress:
    bar.next()

  if debug == True:
    fig = plt.figure()
    spec_plot(fm_mod, fs, fig, sub_plot = (2,2,1), plt_title = 'RF')

  if fc != 0:
    T = len(fm_mod)/fs
    t = linspace(0, T, len(fm_mod))
    I = fm_mod*cos(2*pi*fc*t)
    Q = fm_mod*sin(2*pi*fc*t)

    taps = 265
    edge = (BW/2)/fs
    bands = [0, edge*1.01, edge*1.05, 0.5]
    gains = [1, 0]
    b = remez(taps, bands, gains, type = 'bandpass', maxiter = 1000, 
              grid_density = 32)
    a = 1
    I = lfilter(b, a, I)
    Q = lfilter(b, a, Q)

    if debug == True:
      spec_plot(I, fs, fig, sub_plot = (2,2,2), plt_title = 'I')
      spec_plot(Q, fs, fig, sub_plot = (2,2,3), plt_title = 'Q')

    b = [-1, 1]
    a = 1
    dIdt = lfilter(b, a, I)
    dQdt = lfilter(b, a, Q)
    BB = (I*dQdt - Q*dIdt)/(I**2 + Q**2)

  else: #fc == 0
    b = [-1, 1]
    a = 1
    deriv_fm = lfilter(b, a, fm_mod)
    BB = deriv_fm/sqrt(1 + fm_mod**2)
    """
    integral_msg = np.arccos(fm_mod)
    b = [-1, 1]
    a = 1
    BB = lfilter(b, a, integral_msg)
    """
  if progress:
    bar.next()

  if deemph is True:
    taps = 65
    f1 = 2100.
    f2 = 30000.
    G = f2/f1
    b = remez(taps, [0, f1/fs, f2/fs, 0.5], [1, 1./G], type = 'bandpass',
              maxiter = 100, grid_density = 32)
    a = 1
    BB = lfilter(b, a, BB)

  if debug == True:
    spec_plot(BB, fs, fig, sub_plot = (2,2,4), plt_title = 'BB')
    plt.show()
    
  if progress:
    bar.next()
  mx = max(max(BB), abs(min(BB)))
  BB = BB/mx
  DC = np.average(BB)
  BB = BB - DC

  if progress:
    bar.next()
    bar.finish()
  return BB
