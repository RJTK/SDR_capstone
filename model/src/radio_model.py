from scipy.io import wavfile as wav
from scipy.signal import resample, remez, lfilter
import numpy as np
from numpy import linspace, pi, sqrt, cos, fft
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt

#--------------------------------------------------------------------------------
def spec_plot(x, fs, fig, sub_plot = (1,1,1), plt_title = 'No_Title'):
  '''
  
  x: The time domain spectrum whose spectrum will be plotted
  fs: The sample rate
  fig: The figure to plot on.
  sub_plot: A 3-tuple to indicate the location to plot ie) (1,2,1)
  plt_title: (optional) Title of the plot
  '''
  N = len(x)
  T = N/float(fs)
  f_analog = linspace(-fs/2., fs/2., N)
  
  x_spec = abs(fft.fft(x))
  x_spec = (1./N)*np.append(x_spec[(N-1)/2:],
                         x_spec[0:(N-1)/2])

  ax = fig.add_subplot(*sub_plot) #The * unpacks the tuple
  ax.plot(f_analog, x_spec)
  ax.set_xlabel('freq')
  ax.set_ylabel('|X|/N')
  ax.set_title(plt_title)
  return

#--------------------------------------------------------------------------------
def modulate_fm(x, fsBB, fsRF, fc, del_f = 75000, BW = 200000, A = 1, 
                debug = False):
  '''
  Modulates some signal x with del_f maximum frequency deviation.  The maximum
  message value mp is extracted from x.

  x: The signal to modulate, a 1D np array
  fsBB: The sample rate of the signal x
  fsRF: The sample rate for the modulation
  del_f: delta f, the maximum frequency deviation
  fc: The centre frequency for the modulation
  BW: The final maximum bandwidth of the signal
  A: The amplitude of the output fm modulated signal
  '''
  #Convert everything to float...
  fsBB = float(fsBB)
  fsIF = 3.*BW
  fsRF = float(fsRF)
  fc = float(fc)
  del_f = float(del_f)
  BW = float(BW)
  A = float(A)

  #Perform the modulation, as well as upsampling to fsIF
  T = len(x)/fsBB #The period of time x exists for
  N = fsIF*T #The number of samples for the RF modulation
  m = resample(x, N)
  mp = max(x)
  kf = (2.*pi*del_f)/mp
  t = linspace(0, T, N)
  m_integral = cumtrapz(m, t, initial = 0.)
  fm_modIF = A*cos(kf*m_integral) #Keep the signal at BB

  if debug == True:
    fig = plt.figure()
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,1), plt_title = 'IF')

  #Preemphasis filtering

  taps = 65
  f1 = 2100.
  f2 = 30000.
  G = f2/f1
  b = remez(taps, [0, f1/fsIF, f2/fsIF, 0.5], [1, G], type = 'bandpass',
            maxiter = 100, grid_density = 32)
  a = 1
  fm_modIF = lfilter(b, a, fm_modIF)

  if debug == True:
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,2),
              plt_title = 'IF preemph')
  
  #Bandwidth limiting
  
  right_edge = (BW/2)/fsIF
  b = remez(taps, [0, right_edge*.95, right_edge*.97, 0.5], 
            [1, 0], type = 'bandpass', maxiter = 100, 
            grid_density = 32)
  a = 1
  fm_modIF = lfilter(b, a, fm_modIF)

  if debug == True:
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,3),
              plt_title = 'IF bandlimit')

  #Up sampling and modulating up to fc
  T = len(fm_modIF)/fsIF #The period of time x exists for
  N = fsRF*T #The number of samples for the RF modulation
  t = linspace(0, T*N, N)
  fm_modRF = resample(fm_modIF, N)

  mixer = cos(2*pi*fc*t)
  fm_modRF = mixer*fm_modRF

  if debug == True:
    spec_plot(fm_modRF, fsRF, fig, sub_plot = (2,2,4),
              plt_title = 'RF')
    plt.show()

  return fm_modRF

#--------------------------------------------------------------------------------
def demodulate_fm(fm_modRF, fc, fsRF, fsBB, BW = 200000, debug = False):
  '''
  '''
  fig = plt.figure()
  if debug == True:
    spec_plot(fm_modBB, fsBB, fig, sub_plot = (2,2,4),
              plt_title = 'RF')
    plt.show()

  T = len(fm_modRF)/fsRF #The period of time x exists for
  N = fsBB*T #The number of samples for the RF modulation
  fm_modBB = resample(fm_modRF, N)

  
  
  return

#--------------------------------------------------------------------------------
def main():
  '''
  '''
  fsx = 44100.0
  fsrf = 2000000.0
  fc = 800000.0
  T = 1
  t = linspace(0.0, T, T*fsx)
#  x = (t - sqrt(t))*cos(2*pi*fsx*t)
  x = cos(2*pi*4000*t)
  x_fm = modulate_fm(x, fsx, fsrf, fc, debug = True)

  return

#Program entry point
#===============================================================================
main()
