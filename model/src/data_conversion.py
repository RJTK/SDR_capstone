import numpy as np

from numpy import digitize
from progress.bar import Bar

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
