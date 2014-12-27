import numpy as np
import sys
import os

from scipy.io import wavfile as wav
from scipy.signal import resample
from numpy import sqrt

from modulate_fm import modulate_fm
from channel_models import (AWGN_channel, distortion_channel, multipath_channel,
                            nonlinearphase_channel)
from demodulate_fm import demodulate_fm


#-------------------------------------------------------------------------------
def get_pzero_crossings(x):
  '''
  Returns a vector of boolean values at every positive going zero crossing 
  of the vector x.
  
  x: The input vector
  Returns: A vector of zero crossings...
  '''
  x_pos = x > 0
  x_zx = (x_pos[:-1] & ~x_pos[1:])
  x_zx = np.append(x_zx, 0)
  return x_zx

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

  fm_mod = AWGN_channel(fm_mod, 5, debug = False, progress = show_progress)
  #fm_mod = distortion_channel(fm_mod, taps = 50, progress = show_progress)

  #  fm_mod = multipath_channel(fm_mod, [1.1e-6, 3e-5, 2.6e-5, 2.4e-5, 1e-3],
  #                             [1,1,1,1,1],
  #                             fsIF, debug = True)
  #  fm_mod = nonlinearphase_channel(fm_mod, taps = 12, debug = True)
  #  fm_mod = AWGN_channel(fm_mod, 5, debug = False, progress = show_progress)

  #The analog_to_digital function is only really useful for generating an
  #output file, since the following arithmetic is still done with float64.
  #To model the effects of quantization just add gaussian noise...
  #fm_mod =  analog_to_digital(fm_mod, bits = 4, rnge = [-1,1], file_out = None, 
  #                            progress = show_progress)
  BB = demodulate_fm(fm_mod, fsIF, debug = False, deemph = predeemph, fc = fc,
                     BPL = BPL, progress = show_progress)

  T = len(BB)/fsIF
  N = fsBB*T
  BB = resample(BB, N)

  #  G = sqrt(E_music/E_demod)
  #  print 'Gain: %f' % G
  #How to make volume the same?
  #equating the maximum values does not work
  #Nor does equating the signal energy.
  #The value of G is determined experimentally
  #Also, where in the signal chain is the loss of amplitude even comming from?
  #As far as I can tell, resampling does not effect amplitude.

  E_music = sum(abs(musicRL)**2)
  E_demod = sum(abs(BB)**2)

  G = sqrt(E_music/E_demod)
  BB = G*BB

  musicRL = np.array(musicRL*int16_max, dtype = 'int16')
  BB = np.array(BB*int16_max, dtype = 'int16')
  
#  musicRL = np.array(musicRL, dtype = 'int16')
#  BB = np.array(BB, dtype = 'int16')

  wav.write('radio_broadcast.wav', fsBB, musicRL)
  wav.write('radio_received.wav', fsBB, BB)

#  os.system('aplay radio_broadcast.wav')
  os.system('aplay radio_received.wav')

  return

#Program entry point
#==============================================================================
if __name__ == '__main__':
  main()
