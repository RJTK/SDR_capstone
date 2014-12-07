from numpy import *
from radio_model import *
from progress.bar import Bar

def main():
  '''
  '''
  bar = Bar('Generating data...', max = 9)
  bar.next()
  samp_rate = 1e6
  fc = 3e5
  T = 1
  bits = 12
  t = linspace(0, T, T*samp_rate)
  SNR = 15 #signal to noise ratio
  test_dir = 'test_signals/'

  #Create DC signal at 0.5v
  DC_sig = 0.5*ones(len(t))
  analog_to_digital(DC_sig, bits = bits, file_out = test_dir + 'dc.sig')
  bar.next()

  #Simple sine wave at 5Hz
  cos5_sig = cos(2*5*pi*t)
  analog_to_digital(cos5_sig, bits = bits, file_out = test_dir + 'cos5.sig')
  bar.next()

  #Simple sine wave at 50KHz
  cos50k_sig = cos(2*(50e3)*pi*t)
  analog_to_digital(cos50k_sig, bits = bits, file_out = test_dir + 'cos50k.sig')
  bar.next()

  #Modulate the 50KHz
  cos50kfm_sig, kf = modulate_fm(cos50k_sig, samp_rate, samp_rate)
  analog_to_digital(cos50kfm_sig, bits = bits, 
                    file_out = test_dir + 'cos50kfm.sig')
  bar.next()

  #Music Test file
  fsBB, music = wav.read('classical1.wav')
  music = music[0:2*fsBB] #Truncate to 2 seconds
  musicL = music.T[0] #Separate out the left/right channels
  musicR = music.T[1]
  musicRL = musicL + musicR #merge the two channels
  
  int16_max = float(2**15 - 1)
  musicRL = musicRL/int16_max #normalize to 1

  #Modulated, no preemphasis.  Simplest one
  fm_mod, kf = modulate_fm(musicRL, fsBB, samp_rate, debug = False, 
                           preemph = False, fc = fc)
  analog_to_digital(fm_mod, bits = bits, file_out = test_dir + 'song_simple.sig')
  bar.next()
  
  #Modulated, with preemphasis.
  fm_mod, kf = modulate_fm(musicRL, fsBB, samp_rate, debug = False, 
                           preemph = True, fc = fc)
  analog_to_digital(fm_mod, bits = bits, file_out = test_dir + 'song_preemph.sig')
  bar.next()  

  #Add noise
  fm_mod_noise = AWGN_channel(fm_mod, SNR)
  analog_to_digital(fm_mod_noise, bits = bits, 
                    file_out = test_dir + 'song_noisy.sig')
  bar.next()

  #Add some distortion
  fm_mod_distortion = distortion_channel(fm_mod_noise, taps = 12, debug = True)
  analog_to_digital(fm_mod_distortion, bits = bits, 
                    file_out = test_dir + 'song_distorted.sig')
  bar.next()
  bar.finish()
  return

if __name__ == '__main__':
  main()
