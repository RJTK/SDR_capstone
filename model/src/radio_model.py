from scipy.io import wavfile as wav
from scipy.signal import resample, remez, lfilter, freqz
from progress.bar import Bar
from numpy import linspace, pi, sqrt, cos, sin, fft, append, zeros, digitize
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt
from matplotlib import patches
from numpy.polynomial import polynomial as poly

import numpy as np
import random
import sys
import os


#-------------------------------------------------------------------------------
def spec_plot(x, fs, fig, sub_plot = (1,1,1), plt_title = 'No_Title'):
  '''
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
def demodulate_fm(fm_mod, fs, BW = 200000,
                  debug = False, deemph = True, fc = 0,
                  BPL = True, progress = False):
  '''
  DOCUMENT THIS
  CURRENTLY ONLY EXPERIMENTAL
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

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
def AWGN_channel(x, SNR, debug = False, progress = False):
  '''
  Adds additive white Gaussian noise to the signal contained in x.  Calculations
  are preformed to ensure the resulting signal to noise ratio is specified
  by the parameter 'SNR'

  x: The signal to add noise to
  SNR: The signal to noise ratio in dB
  '''
  if progress:
    bar = Bar('Adding AWGN...', max = 4)
    bar.next()
  SNR = float(SNR)
  Px = sum(x**2)/len(x) #Energy of x
  PN = Px/(10**(SNR/10))
  var = sqrt(PN)
  if progress:
    bar.next()
  AWGN = np.random.normal(0, var, len(x))

  if progress:
    bar.next()
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
  mx = max(max(x + AWGN), abs(min(x + AWGN)))
  noisy_signal = (x + AWGN)/mx

  if progress:
    bar.next()
    bar.finish()
  return noisy_signal

#-------------------------------------------------------------------------------
def distortion_channel(x, tap_magnitude = 1, taps = 5, debug = False, 
                       progress = False):
  '''
  Adds extreme distortion to the signal x.  This is not a realistic channel,
  but serves for a 'worst case scenario' test.  Applies an FIR filter with
  random taps.  Note that using an IIR filter with random taps will be
  unstable.
  
  x: The signal to be distorted
  tap_magnitude: The maximum magnitude (+/-) of the taps
  taps: The number of FIR filter taps
  '''
  if progress:
    bar = Bar('Applying distortive filter...', max = 3)
    bar.next()
  b = map(random.uniform, [-1*tap_magnitude]*taps,
          [tap_magnitude]*taps) #generates a random array

  if progress:
    bar.next()
  a = 1
  distorted = lfilter(b, a, x)
  mx = max(max(distorted), -1*min(distorted))
  distorted = distorted/mx #re-normalize to unity magnitude

  if progress:
    bar.next()
  if debug == True:
    plot_filter(b,a)
    fig = plt.figure()
    spec_plot(x, 800000, fig, sub_plot = (1,2,1), plt_title = 'Original')
    spec_plot(distorted, 800000, fig, sub_plot = (1,2,2), plt_title = 'Distorted')
    fig.show()

  if progress:
    bar.finish()
  return distorted

#-------------------------------------------------------------------------------
def multipath_channel(x, delay_time, G, fs, debug = False):
  '''
  Models a multipath channel.  This function creates a delayed version of the 
  signal 'x' by disgarding samples, multiplies by a gain factor G, and adds
  to the original signal.  delay_time & G may both be iterables, this will
  add multiple copies.

  *** Modeling multipath distortion is not an easy task.  The problem is that
  1 sample's worth of time is a long time compared to the speed of light.
  Modeling multipath effects is thus more complicated than just delaying the
  signal by an integral number of samples.  I need a method to shift the
  signal by a fraction of it's sample width.  This is definitely possible
  through some sort of interpolation. ***

  x: The original signal
  delay_time: A delay time in seconds.  That is, the multipath propagation delay.
    This will probably be on the order of micro seconds or nano seconds.  Light
    takes appx 3ns to travel 1m.  So, roughly 3us to travel 1Km
  G: A gain factor for the multipath.  Must be 0 <= G <= 1
  fs: The sample rate (used to calculate time in seconds)
  '''
  bar = Bar('Multipath channel...', max = 3 + len(G))
  bar.next()
  fs = float(fs)
  assert len(delay_time) == len(G)
  G = [float(g) for g in G]
  delay_time = [float(dt) for dt in delay_time]
  for g in G:
    assert(0 <= g and g <= 1)
  multipath_signals = []
  bar.next()

  dt_sample = 1./fs #Amount of time per sample  
  for T, g in zip(delay_time, G):
    n_samples = T/dt_sample #Number of samples for the delay time
    n_samples = int(round(n_samples)) #Round to nearest sample
    if n_samples <= 0:
      print '  WARN: 0 samples of delay'
    if debug == True:
      print '  Delaying by %d samples' % n_samples
    multipath = x[0:-1*n_samples]
    multipath = append(zeros(n_samples), multipath)
    multipath = multipath*g
    multipath_signals.append(multipath)
    bar.next()

  y = x
  for m in multipath_signals:
    y = y + m
  mx = max(max(y), -1*min(y))
  y = y/mx #normalize to 1
  bar.next()

  if debug == True:
    fig = plt.figure()
    spec_plot(x, fs, fig, sub_plot = (1,2,1), plt_title = 'Original')
    spec_plot(multipath_signals[0], fs, fig, sub_plot = (1,2,2), 
              plt_title = 'Multipath 1')
    fig.show()

  bar.finish()
  return y

#-------------------------------------------------------------------------------
def nonlinearphase_channel(x, taps = 5, debug = False):
  '''
  This does not actually do what I hoped it would.  I expected a terribly ugly
  phase response.  However, the phase response of this random IIR filter turns
  out to actually be pretty linear...
  '''
  bar = Bar('Passing through nonlinearphase channel...', max = taps + 1)
  bar.next()

  rnge = (0.2, 0.80)
  P = [] #Poles
  if taps % 2 == 0:
    pass
  else:
    taps = taps - 1
    p = random.uniform(*rnge)
    p = random.choice([-1,1])*p
    P.append(p)
  for i in range(taps): 
    p_real = random.uniform(*rnge)
    p_real = random.choice([-1, 1])*p_real
    p_imag = random.uniform(0.2, 0.80*sqrt(1 - p_real**2))
    p_imag = random.choice([-1, 1])*p_imag

    p1 = p_real + (1j)*p_imag #1j is an imaginary number
    p2 = p_real + (1j)*(-1)*p_imag
    P.append(p1)
    P.append(p2)
    bar.next()
    
  a = poly.polyfromroots(P) #a is the 'a' coefficients of the IIR filter

  b = np.conj(a) #b must be the reversed & conjugated coefficients of a
  b = np.array(list(reversed(b)))

  #The polynomial library stores it's coefficients in the opposite order
  #of scipy.signal.  So i reverse both here again.
  #Reversing a twice brings it back to it's original form, but I am hopefully
  #avoiding some confusion because it is being done for 2 different reasons.
  b = np.array(list(reversed(b)))
  a = np.array(list(reversed(a)))

  if debug == True:
    plot_filter(b,a)
    plot_filter_pz(b,a)
  
  x = lfilter(b, a, x)
  max_ampl = max(max(x), -1*min(x))
  x = x/max_ampl

  bar.next()
  bar.finish()
  return x

#-------------------------------------------------------------------------------
def analog_to_digital(x, bits = 12, rnge = [-1,1], file_out = None, 
                      progress = False):
  '''
  Uses numpy.digitize to simulate an analog to digital converter.  The rnge
  argument needs to be a list specifying the minimum and maximum values of the
  input.  The 'bits' argument specifies the number of bits to use in the
  quantization.  If file_out is not None, it will be interpreted as a name of
  a file, and the digitized signal will be written to the file, one sample per
  line, in 2's complement binary notation.
  '''
  if len(rnge) != 2:
    raise AssertionError('rnge must have a length of 2')
  if progress:
    bar = Bar('%d bit ADC...' % bits, max = 4)
    bar.next()
  FSR = float(rnge[1] - rnge[0]) #Full Scale Range
  inc = FSR/(2**bits - 1) #Increment size
  bins = np.array([rnge[0] + i*inc for i in range(2**bits - 1)]) #Bin boundaries
  if progress:
    bar.next()
  quantizedb = digitize(x, bins) - 2**(bits - 1)
  #We discard the -FS value
  quantizedb = np.array([-2**(bits - 1) + 1 if q == -2**(bits - 1) else 
                         q for q in quantizedb])
  if progress:
    bar.next()
  #Convert to float and normalize
  quantizedf = np.array(quantizedb, dtype = np.float64)
  quantizedf = quantizedf/(2**(bits - 1))
  if file_out != None:
    #*** I have realized that this is kind of stupid.  The FPGA doesn't want
    #*** to read strings.  It wants 12b words.  How to store that in a file?
    #*** And then, how to send that to the FPGA?
    f = open(file_out, 'w')
    #quantizedb is the version expressed in bits
    for q in quantizedb:
      f.write(np.binary_repr(q, bits - 1))
      f.write('\n')
    f.close()
  if progress:
    bar.next()
    bar.finish()
  return quantizedf

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
