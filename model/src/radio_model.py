import numpy as np
import sys
import os

from scipy.io import wavfile as wav
from scipy.signal import resample
from numpy import sqrt

from modulate_fm import modulate_fm
from channel_models import (AWGN_channel, distortion_channel, multipath_channel,
                            nonlinearphase_channel)
from IQ_demodulate_fm import iqdemodulate_fm
from ZX_demodulate import zxdemodulate_fm

from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------
def main():
  '''
  '''
  sys.setcheckinterval(1000000)
  try:
    os.nice(-20)
  except OSError:
    print 'Note, running as root sets this script to a higher priority'

  show_progress = True
  
  print 'Reading file...'
  fsBB, music = wav.read('classical1.wav')
  music = music[0:2*fsBB] #Truncate to 2 seconds
  musicL = music.T[0] #Separate out the left/right channels
  musicR = music.T[1]
  musicRL = musicL + musicR #merge the two channels
  
  int16_max = float(2**15 - 1)
  musicRL = musicRL/int16_max #normalize to 1

  fsIF = 800000. 
  fc = 100000
  BPL = True #Bandpass limiter, see demodulator
  predeemph = True #Preemphasis and deemphasis filtering

  fm_mod, kf = modulate_fm(musicRL, fsBB, fsIF, debug = False, 
                           preemph = predeemph, fc = fc, progress = show_progress)

  T = len(fm_mod)/fsIF
  t = np.linspace(0, T, T*fsIF)
  plt.plot(t[0:10000], fm_mod[0:10000])
  plt.show()
  return
#  fm_mod = AWGN_channel(fm_mod, 10, debug = False, progress = show_progress)
#  fm_mod = distortion_channel(fm_mod, taps = 10, progress = show_progress)

#  fm_mod = multipath_channel(fm_mod, [1e-7],
#                               [1], fsIF, debug = False)
#  fm_mod = nonlinearphase_channel(fm_mod, taps = 12, debug = True)
  fm_mod = AWGN_channel(fm_mod, 0, debug = False, progress = show_progress)

  #The analog_to_digital function is only really useful for generating an
  #output file, since the following arithmetic is still done with float64.
  #To model the effects of quantization just add gaussian noise...
  #fm_mod =  analog_to_digital(fm_mod, bits = 4, rnge = [-1,1], file_out = None, 
  #                            progress = show_progress)
  BB_IQ = iqdemodulate_fm(fm_mod, fsIF, debug = False, deemph = predeemph, fc = fc,
                          BPL = BPL, progress = show_progress)
  BB_ZX = zxdemodulate_fm(fm_mod, fsIF, debug = False, deemph = predeemph, fc = fc,
                          BPL = BPL, progress = show_progress)

  play(BB_IQ, fsIF, fsBB, norm = 'inf', debug = False,
       msg = 'Playing IQ, inf norm')
  play(BB_ZX, fsIF, fsBB, norm = 'inf', debug = False,
       msg = 'Playing ZX, inf norm')
  return

def play(x, fsIF, fsBB, norm = 'inf', debug = False, msg = False):
  '''
  '''
  norm = 'inf' #Only available
  if msg:
    print msg

  int16_max = float(2**15 - 1)
  T = len(x)/fsIF
  N = fsBB*T
  x = resample(x, N)
  
  norm_inf = max(max(x), abs(min(x)))

  if norm == 'inf':
    x = x/norm_inf
  else:
    raise ValueError('norm must be \'inf\'')

  if debug:
    print 'The pre norm max value of x is %f' % norm_inf
    print 'The post norm max value of x is %f' % max(max(x), abs(min(x)))

  x = np.array(x*int16_max, dtype = 'int16')
  wav.write('play_me.wav', fsBB, x)
  os.system('aplay play_me.wav')

  return

#Program entry point
#==============================================================================
if __name__ == '__main__':
  main()
