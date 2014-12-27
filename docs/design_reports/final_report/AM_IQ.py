from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace, sin, cos, pi, sqrt, concatenate, array

def main():
  phase_mod()
  return

def phase_mod():
  t = linspace(0, 1, 1000)
  lt = len(t)/4
  PSK = concatenate(([1]*lt, [-1]*lt, [1]*lt, [-1]*lt))

  I = cos(2*pi*t*10)
  Q = sin(2*pi*t*10*PSK)

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection = '3d')
  ax.set_xlabel('t')
  ax.set_ylabel('I')
  ax.set_zlabel('Q')

  ax.plot(t, I, Q)
  plt.show()
  return

def AM_mod():
  t = linspace(0, 1, 1000)
  Imod = 0.5 + 0.5*cos(2*pi*t*3)
  Qmod = 0.5 + 0.5*cos(2*pi*t*3)
  
  I = Imod*cos(2*pi*t*50)
  Q = Qmod*sin(2*pi*t*50)

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1, projection = '3d')
  ax.set_xlabel('t')
  ax.set_ylabel('I')
  ax.set_zlabel('Q')

  ax.plot(t, I, Q)
  plt.show()
  return

if __name__ == '__main__':
  main()
