import numpy as np

from numpy import pi
from scipy.signal import remez, lfilter
from matplotlib import pyplot as plt

from spec_plot import spec_plot

def bandpass_limiter(x, BW, fc, fs, debug = False):
  '''
  '''
  limited = [1 if i >= 0 else -1 for i in x]
  
  DC = np.average(limited)
  limited = limited - DC

  left_edge = (fc - (BW/2.))/fs
  right_edge = (fc + (BW/2.))/fs
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
  const_amplx = (pi/4)*lfilter(b, a, limited)
  mx = max(max(const_amplx), abs(min(const_amplx)))
  const_amplx = const_amplx/mx #normalize to 1

  if debug == True:
    fig = plt.figure()
    spec_plot(x, fs, fig, sub_plot = (2,2,1), plt_title = 'original')
    spec_plot(limited, fs, fig, sub_plot = (2,2,2), plt_title = 'limited')
    spec_plot(const_amplx, fs, fig, sub_plot = (2,2,3), plt_title = 'BPL')
    fig.show()

  return const_amplx
