import numpy as np

from scipy.signal import freqz
from matplotlib import pyplot as plt
from matplotlib import patches

def plot_filter_pz(b, a):
  '''
  Creates a pole zero plot of a filter
  Modified from: http://www.dsprelated.com/showcode/244.php
  '''
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  uc = patches.Circle((0, 0), radius = 1, fill = False,
                      color = 'black', ls = 'dashed')
  ax.add_patch(uc)

  z = np.roots(b)
  p = np.roots(a)

  ax.plot(z.real, z.imag, 'go', linewidth = 2)
  ax.plot(p.real, p.imag, 'rx', linewidth = 2)

  fig.show()
  return

#-------------------------------------------------------------------------------
def plot_filter(b, a):
  '''
  Plots a filter's response
  '''
  w, h = freqz(b, a)
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  ax1.set_title('frequency response')

  ax1.plot(w, abs(h), 'b')
  ax1.set_ylabel('|H|', color = 'b')
  ax1.set_xlabel('frequency [Rad/Sample]')

  ax2 = ax1.twinx()
  angles = np.unwrap(np.angle(h))
  ax2.plot(w, angles, 'g')
  ax2.set_ylabel('Angle (radians)', color = 'g')
  ax2.grid()
  fig.show()
  return

def main():
  plot_filter([1, -1], [1])

if __name__ == '__main__':
  main()
  while True:
    pass
