'''
A grouping of a few channel models I made.  These are not meant to be realistic
models.  Only to help me look at some worst-case scenarios.
'''

import numpy as np

from progress.bar import Bar
from numpy import sqrt, random, zeros, append
from scipy.signal import lfilter
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as poly

from spec_plot import spec_plot
from filter_plots import plot_filter_pz, plot_filter

def AWGN_channel(x, SNR, debug = False, progress = False):
  '''
  *WELL TESTED*
  
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
  *WELL TESTED*

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
  *NOT COMPLETE*

  Models a multipath channel.  This function creates a delayed version of the 
  signal 'x' by disgarding samples, multiplies by a gain factor G, and adds
  to the original signal.  delay_time & G may both be iterables, this will
  add multiple copies.

  *** Modeling multipath distortion is not an easy task.  The problem is that
  1 sample's worth of time is a long time compared to the speed of light.
  Modeling multipath effects is thus more complicated than just delaying the
  signal by an integral number of samples.  I need a method to shift the
  signal by a fraction of it's sample width.  This can be achieved by massive
  upsampling.  However, this approach is computationally expensive.  Is it
  possible to effectively shift the signal by some small amount, without
  incurring a huge computational cost? ***

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
  *SOME TESTING, BUGGY*

  Implements an all pass filter (flat frequency response).  The plan was to
  generate an all pass filter with a very ugly phase response by generating
  random taps (while maintaining flat frequency response).  However, this does
  not actually do what I hoped it would.  I expected a terribly ugly
  phase response.  However, the phase response of this random IIR filter turns
  out to actually be pretty linear...

  The frequency response seems to also SOMTIMES output some crazy behaviour as 
  the frequency gets close to pi.  Would be wise to use with debug = True.
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
    p_imag = random.uniform (0.2, 0.80*sqrt(1 - p_real**2))
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
