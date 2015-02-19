from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace, sin, cos, pi, sqrt, concatenate, array, log10
from scipy.integrate import cumtrapz
from scipy.signal import remez, lfilter, resample, freqz, firwin2
import numpy as np
import os, sys

#---------------------------------------------------------------------------------
def main():
  filter_examples()
  return

#---------------------------------------------------------------------------------
def zero_crossing_example():
  '''
  Generates plots for zero crossing detector
  '''
  path = os.path.abspath(os.path.join('..', '..', '..', 'model', 'src'))
  sys.path.append(path)
  from ZX_demodulate import get_pzero_crossings
  from modulate_fm import modulate_fm
  from channel_models import AWGN_channel

  x = freq_mod(False)
  t = linspace(0, 1, 1000)
  m = sin(2*pi*5*t)
  x = resample(x, 1000)
  
  taps = 50
  edge = 0.01
  bands = [0, edge*.97, edge*1.03, 0.5]
  gains = [1, 0]
  b = remez(taps, bands, gains)

  pulse_width = 10
  pulses = np.ones(pulse_width)

  zx = get_pzero_crossings(x)
  zxp = np.convolve(zx, pulses)[pulse_width - 1:]
  zxp = zxp - np.average(zxp)
  bb = lfilter(b, [1], zxp)
  bb = lfilter(b, [1], bb)
  bb = lfilter(b, [1], bb)
  bb = lfilter(b, [1], bb)
  mx = max(max(bb), abs(min(bb)))
  bb = bb/mx

  fig = plt.figure()
  t = linspace(0, 1, len(x))

  ax = fig.add_subplot(4, 1, 1)
  ax.set_xlabel('$t$')
  ax.set_ylabel('FM Modulated')
  ax.plot(t, x, label = '$\psi_{FM}(t)$')
  ax.plot(t, m, 'r', label = 'BB')

  ax = fig.add_subplot(4, 1, 2)
  ax.set_xlabel('$t$')
  ax.set_ylabel('Zero Crossings')
  ax.plot(t, zx, label = 'Zero Xings')

  ax = fig.add_subplot(4, 1, 3)
  ax.set_xlabel('$t$')
  ax.set_ylabel('Pulses')
  ax.plot(t, zxp, label = 'Pulsed Zero Xings')

  ax = fig.add_subplot(4, 1, 4)
  ax.set_xlabel('$t$')
  ax.set_ylabel('BaseBand')
  ax.plot(t, bb, label = 'Baseband')
  ax.plot(t, m, 'r', label = 'BB original')

  fig.show()
  x = raw_input('Continue...')
  return

#---------------------------------------------------------------------------------
def freq_mod(plot = True):
  '''
  Generates a plot for a frequency modulated signal.  Also returns a
  good example signal.
  '''
  f_m = 5.
  f_c = 50.
  kf = 200.
  t = linspace(0, 1, 50000)
  m = sin(2*pi*f_m*t)
  
  fm = cos(2*pi*t*f_c + kf*cumtrapz(m, dx = 1./len(m), initial = 0.))

  if plot:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Signal Strength')
    ax.plot(t, fm, label = '$\psi_{FM}(t)$')
    ax.plot(t, m, 'r', linewidth = 2, label = '$m(t)$')
    ax.legend()
    plt.show()
  return fm

#---------------------------------------------------------------------------------
def phase_mod():
  '''
  IQ space plot of PSK
  '''
  t = linspace(0, 1, 1000)
  lt = len(t)/4
  PSK = concatenate(([1]*lt, [-1]*lt, [1]*lt, [-1]*lt))

  I = cos(2*pi*t*10)
  Q = sin(2*pi*t*10*PSK)

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection = '3d')
  ax.set_xlabel('t')
  ax.set_ylabel('I')
  ax.set_zlabel('Q')

  ax.plot(t, I, Q)
  plt.show()
  return

#---------------------------------------------------------------------------------
def AM_mod():
  '''
  IQ space plot of AM
  '''
  t = linspace(0, 1, 1000)
  Imod = 0.5 + 0.5*cos(2*pi*t*3)
  Qmod = 0.5 + 0.5*cos(2*pi*t*3)
  
  I = Imod*cos(2*pi*t*50)
  Q = Qmod*sin(2*pi*t*50)

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection = '3d')
  ax.set_xlabel('t')
  ax.set_ylabel('I')
  ax.set_zlabel('Q')

  ax.plot(t, I, Q)
  plt.show()
  return

#---------------------------------------------------------------------------------
def filter_examples():
  '''
  Plots some example bandpass filters with different numbers of taps
  '''
  bands = [0, .19, .21, .29, .31, .5]
  gains = [0, 1, 0]
#  bands = [0, .19, .2, .3, .31, .5]
#  gains = [0, 0, 1, 1, 0, 0]
  B = []
  a = 1
  taps = range(10, 131, 30)
  for t in taps:
    B.append(remez(t, bands, gains, type = 'bandpass', maxiter = 1000,
                   grid_density = 32))

#    B.append(firwin2(t, bands, gains, nyq = 0.5))
    
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  ax1.set_title('Frequency Responses for Various FIR lengths')
  ax1.set_ylabel('Amplitude [dBV]')
  ax1.set_xlabel('Frequency [Rad/Sample]')
  for i, b in enumerate(B):
    w, h = freqz(b, a)
    ax1.plot(w, 20*log10(abs(h)), label = '%d taps' % taps[i])
  ax1.legend()
  fig.show()
  raw_input('Continue?...')
  return
                 

#---------------------------------------------------------------------------------
if __name__ == '__main__':
  main()
