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
def modulate_fm(x, fsx, fsrf, fc, del_f = 75000, BW = 200000, A = 1):
  '''
  Modulates some signal x with del_f maximum frequency deviation.  The maximum
  message value mp is extracted from x.

  x: The signal to modulate, a 1D np array
  fsx: The sample rate of the signal x
  fsrf: The sample rate for the modulation
  del_f: delta f, the maximum frequency deviation
  fc: The centre frequency for the modulation
  BW: The final maximum bandwidth of the signal
  A: The amplitude of the output fm modulated signal
  '''
  #Convert everything to float...
  fsx = float(fsx)
  fsrf = float(fsrf)
  fc = float(fc)
  del_f = float(del_f)
  BW = float(BW)
  A = float(A)

  #Perform the modulation, as well as upsampling
  T = len(x)/fsx #The period of time x exists for
  N = fsrf*T #The number of samples for the RF modulation
  assert(fsrf > 2*fc), 'The sample rate must be at least twice the RF frequency'
  assert(fsrf >= fsx), 'The RF sample rate must be higher than the BB sample rate'
  m = resample(x, N)
  mp = max(x)
  kf = (2.*pi*del_f)/mp
  t = linspace(0, T, N)
  m_integral = cumtrapz(m, t, initial = 0.)
  fm_mod = A*cos(2.*pi*fc*t + kf*m_integral)

  #---------------------------------------------
  #I DON'T KNOW IF THIS IS CORRECT CHECK WORK
  #I used freqz to plot response, it is questionable
  #Preemphasis
  prd = 1/fsrf #T is already taken
  #f1 and f2 are standard values for FM preemphasis
  f1 = 2100
  f2 = 30000
  #a and b derived from bilinear transform
  b = [f2*2*(pi*f1*prd + 1), f2*2*(pi*f1*prd - 1)]
  a = [f1*2*(pi*f1*prd + 1), f1*2*(pi*f2*prd - 1)]
  fm_mod = lfilter(b, a, fm_mod)
  #---------------------------------------------

  #Bandwidth limiting
  left_edge = (fc - (BW/2))/fsrf
  right_edge = (fc + (BW/2))/fsrf
  taps = 65
  b = remez(taps, [0, left_edge*1.03, left_edge*1.05, 
                   right_edge*.95, right_edge*.97, 0.5], 
            [0, 1, 0], type = 'bandpass', maxiter = 100, 
            grid_density = 32)
  a = 1
  fm_mod = lfilter(b, a, fm_mod)

  return fm_mod

#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
def main():
  '''
  '''
  fsx = 44100.0
  fsrf = 2000000.0
  fc = 400000.0
  T = 1
  t = linspace(0.0, T, T*fsx)
#  x = (t - sqrt(t))*cos(2*pi*fsx*t)
  x = cos(2*pi*4000*t)
  x_fm = modulate_fm(x, fsx, fsrf, fc)

  fig = plt.figure()
  spec_plot(x, fsx, fig, sub_plot = (1,2,1), plt_title = 'Base Band')
  spec_plot(x_fm, fsrf, fig, sub_plot = (1,2,2), plt_title = 'FM Modulation')
  plt.show()

  return

#Program entry point
#===============================================================================
main()
