import numpy as np

from progress.bar import Bar
from scipy.signal import resample, remez, lfilter
from matplotlib import pyplot as plt
from numpy import pi, linspace, cos
from scipy.integrate import cumtrapz

from spec_plot import spec_plot

def modulate_fm(x, fsBB, fsIF, del_f = 75000, BB_BW = 15000, 
                BW = 200000, A = 1, debug = False, 
                preemph = True, fc = 0, progress = False):
  '''
  SEEMS TO WORK ALRIGHT

  Modulates some signal x with del_f maximum frequency deviation.  The maximum
  message value mp is extracted from x.

  x: The signal to modulate, a 1D np array
  fsBB: The sample rate of the signal x
  fsRF: The sample rate for the modulation
  del_f: delta f, the maximum frequency deviation
  fc: The centre frequency for the modulation
  BW: The final maximum bandwidth of the signal
  A: The amplitude of the output fm modulated signal
  Returns: An fm modulated signal
  '''
  #Convert everything to float...
  fsBB = float(fsBB)
  fsIF = float(fsIF)
  del_f = float(del_f)
  BW = float(BW)
  A = float(A)

  if progress:
    bar = Bar('FM Modulating ...', max = 6)
    bar.next()

  taps = 65
  right_edge = BB_BW/fsBB
  b = remez(taps, [0, right_edge*.95, right_edge*.97, 0.5], 
            [1, 0], type = 'bandpass', maxiter = 100, 
            grid_density = 32)
  a = 1
  BB = lfilter(b, a, x)
  if progress:
    bar.next()

  if debug == True:
    fig = plt.figure()
    spec_plot(BB, fsBB, fig, sub_plot = (2,2,1), plt_title = 'BB')

  #Perform the modulation, as well as upsampling to fsIF
  T = len(BB)/fsBB #The period of time x exists for
  N = fsIF*T #The number of samples for the RF modulation
  if not fsBB == fsIF:
    BB = resample(BB, N)
  mp = max(BB)
  kf = (2.*pi*del_f)/mp
  if progress:
    bar.next()

  #Preemphasis filtering
  if preemph is True:
    taps = 65
    f1 = 2100.
    f2 = 30000.
    G = f2/f1
    b = remez(taps, [0, f1/fsIF, f2/fsIF, 0.5], [1, G], type = 'bandpass',
              maxiter = 100, grid_density = 32)
    a = 1
    BB = lfilter(b, a, BB)

    if debug == True:
      spec_plot(BB, fsIF, fig, sub_plot = (2,2,2),
                plt_title = 'Preemphasized BB')
  
  if progress:
    bar.next()
  #FM modulation
  t = linspace(0, T, len(BB))
  BB_integral = cumtrapz(BB, dx = 1./len(BB), initial = 0.)
  fm_modIF = A*cos(2*pi*fc*t + kf*BB_integral)

  DC = np.average(fm_modIF)
  fm_modIF = fm_modIF - DC

  if debug == True:
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,3), plt_title = 'Modulated')
  if progress:
    bar.next()

  #Bandwidth limiting
  left_edge = (fc - (BW/2.))/fsIF
  right_edge = (fc + (BW/2.))/fsIF
  taps = 165
  if left_edge <= 0:
    if right_edge == 0.5:
      bands = [0,0.5]
      gains = [1]
    bands = [0, right_edge*.97, right_edge*.99, 0.5]
    gains = [1, 0]
  elif right_edge == 0.5:
    bands = [0, left_edge*1.01, left_edge*1.03, 0.5]
    gains = [0, 1]
  else:
    bands = [0, left_edge*1.01, left_edge*1.05,
             right_edge*.95, right_edge*.99, 0.5]
    gains = [0, 1, 0]

  b = remez(taps, bands, gains, type = 'bandpass', maxiter = 1000, 
            grid_density = 32)
  a = 1
  fm_modIF = lfilter(b, a, fm_modIF)
  if progress:
    bar.next()

  if debug == True:
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,4),
              plt_title = 'Transmitted')
    fig.show()

  mx = max(max(fm_modIF), abs(min(fm_modIF)))
  fm_modIF = fm_modIF/mx #normalize to 1
  if progress:
    bar.finish()
  return fm_modIF, kf

#-------------------------------------------------------------------------------    
