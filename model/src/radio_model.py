from scipy.io import wavfile as wav
from scipy.signal import resample, remez, lfilter, hilbert, freqz
import numpy as np
from numpy import linspace, pi, sqrt, cos, sin, fft
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
import sys
import os

#-------------------------------------------------------------------------------
def spec_plot(x, fs, fig, sub_plot = (1,1,1), plt_title = 'No_Title'):
  '''
  FAIRLY WELL TESTED

  x: The time domain spectrum whose spectrum will be plotted
  fs: The sample rate
  fig: The figure to plot on.
  sub_plot: A 3-tuple to indicate the location to plot ie) (1,2,1)
  plt_title: (optional) Title of the plot
  '''
  N = len(x)
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

#-------------------------------------------------------------------------------
def modulate_fm(x, fsBB, fsIF, del_f = 75000, BB_BW = 15000, 
                BW = 200000, A = 10, debug = False, 
                preemph = True, fc = 0):
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

  print 'Performing Baseband Bandwidth Limiting...'
  taps = 65
  right_edge = BB_BW/fsBB
  b = remez(taps, [0, right_edge*.95, right_edge*.97, 0.5], 
            [1, 0], type = 'bandpass', maxiter = 100, 
            grid_density = 32)
  a = 1
  BB = lfilter(b, a, x)

  if debug == True:
    fig = plt.figure()
    spec_plot(BB, fsBB, fig, sub_plot = (2,2,1), plt_title = 'BB')

  #Perform the modulation, as well as upsampling to fsIF
  print 'upsampling to IF...'
  T = len(BB)/fsBB #The period of time x exists for
  N = fsIF*T #The number of samples for the RF modulation
  BB = resample(BB, N)
  mp = max(BB)
  kf = (2.*pi*del_f)/mp

  #Preemphasis filtering
  if preemph is True:
    print 'Performing Preemphasis Filtering...'

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
  
  #FM modulation
  print 'FM Modulating...'

  t = linspace(0, T, len(BB))
  BB_integral = cumtrapz(BB, dx = 1./len(BB), initial = 0.)
  fm_modIF = A*cos(2*pi*fc*t + kf*BB_integral)

  DC = np.average(fm_modIF)
  fm_modIF = fm_modIF - DC

  if debug == True:
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,3), plt_title = 'Modulated')

  #Bandwidth limiting

  print 'Performing Modulated Bandwidth Limiting...'

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

  if debug == True:
    spec_plot(fm_modIF, fsIF, fig, sub_plot = (2,2,4),
              plt_title = 'Transmitted')
    fig.show()

  return fm_modIF, kf

#-------------------------------------------------------------------------------
def demodulate_fm(fm_mod, fs, BW = 200000,
                  debug = False, deemph = True, fc = 0,
                  BPL = True):
  '''
  DOCUMENT THIS
  CURRENTLY ONLY EXPERIMENTAL
  '''
  fs = float(fs)
  BW = float(BW)

  fm_mod = bandpass_limiter(fm_mod, BW, fc, fs, debug)

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
    I = lfilter(b, 1, I)
    Q = lfilter(b, 1, Q)

    if debug == True:
      spec_plot(I, fs, fig, sub_plot = (2,2,2), plt_title = 'I')
      spec_plot(Q, fs, fig, sub_plot = (2,2,3), plt_title = 'Q')

    b = [-1, 1]
    a = 1
    dIdt = lfilter(b, a, I)
    dQdt = lfilter(b, a, Q)
    BB = (I*dQdt - Q*dIdt)/(I**2 + Q**2)

  else: #fc == 0
    integral_msg = np.arccos(fm_mod)
    b = [-1, 1]
    a = 1
    BB = lfilter(b, a, integral_msg)

  if deemph is True:
    print 'Performing deemphasis Filtering...'

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
    
  return BB

#-------------------------------------------------------------------------------
def get_pzero_crossings(x):
  '''
  Returns a vector of boolean values at every positive going zero crossing 
  of the vector x.
  
  x: The input vector
  Returns: A vector of zero crossings...
  '''
  N = len(x)
  x_pos = x > 0
  x_zx = (x_pos[:-1] & ~x_pos[1:])
  x_zx = np.append(x_zx, 0)
  return x_zx

#-------------------------------------------------------------------------------
def bandpass_limiter(x, BW, fc, fs, debug = False):
  '''
  '''
  limited = [1 if i > 0 else -1 for i in x]
  
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

#-------------------------------------------------------------------------------
def plot_filter(b, a):
  '''
  Plots a filter's response
  '''
  w, h = freqz(b, a)
  fig = plt.figure()
  plt.title('frequency response')
  ax1 = fig.add_subplot(1,1,1)

  plt.plot(w, abs(h), 'b')
  plt.ylabel('|H|', color = 'b')
  plt.xlabel('frequency [Rad/Sample]')

  ax2 = ax1.twinx()
  angles = np.unwrap(np.angle(h))
  plt.plot(w, angles, 'g')
  plt.ylabel('Angle (radians)', color = 'g')
  plt.grid()
  plt.show()
  return

#-------------------------------------------------------------------------------
def AWGN_channel(x, SNR, debug = False):
  '''
  x: The signal to add noise to
  SNR: The signal to noise ratio in dB
  '''
  SNR = float(SNR)
  Px = sum(x**2)/len(x) #Energy of x
  PN = Px/(10**(SNR/10))
  var = sqrt(PN)
  AWGN = np.random.normal(0, var, len(x))
  if debug == True:
    print 'len x = %d' % len(x)
    EPx = Px
    EPN = sum(AWGN**2)/len(AWGN)
    SNR = EPx/EPN
    SNRdB = 10*np.log10(SNR)
    print 'AWGN Channel...'
    print 'Signal power: %f' % EPx
    print 'Noise power: %f' % EPN
    print 'SNR (not dB): %f' % SNR
    print 'SNR: %fdB' % SNRdB
  return x + AWGN

#-------------------------------------------------------------------------------
def main():
  '''
  '''
  sys.setcheckinterval(1000000)
  try:
    os.nice(-20)
  except OSError:
    print 'Note, running as root sets this script to a higher priority'

  print 'Reading file...'
  fsBB, music = wav.read('classical1.wav')
  music = music[0:2*fsBB] #Truncate to 2 seconds
  musicL = music.T[0] #Separate out the left/right channels
  musicR = music.T[1]
  musicRL = musicL + musicR #merge the two channels
  
  int16_max = float(2**15 - 1)
  musicRL = musicRL/int16_max

  fsIF = 800000. 
  fc = 182100
  BPL = True #Bandpass limit
  predeemph = True #Preemphasis and deemphasis filtering
  fm_mod, kf = modulate_fm(musicRL, fsBB, fsIF, debug = False, 
                           preemph = predeemph, fc = fc)
  fm_mod = AWGN_channel(fm_mod, 5, debug = False)
  BB = demodulate_fm(fm_mod, fsIF, debug = False, deemph = predeemph, fc = fc,
                     BPL = BPL)

  DC = np.average(BB)
  BB = BB - DC

  T = len(BB)/fsIF
  N = fsBB*T
  BB = resample(BB, N)

  E_music = sum(abs(musicRL)**2)
  E_demod = sum(abs(BB)**2)
  
  #  G = sqrt(E_music/E_demod)
  #  print 'Gain: %f' % G
  #How to make volume the same?
  #equating the maximum values does not work
  #Nor does equating the signal energy.
  #The value of G is determined experimentally
  #Also, where in the signal chain is the loss of amplitude even comming from?
  #As far as I can tell, resampling does not effect amplitude.

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
