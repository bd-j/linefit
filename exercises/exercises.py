import numpy as np

cols = ['id','x','y','sx','sy','rxy']
types = len(cols) * ['<f8']
dtype = np.dtype( zip(cols, types))

dat = np.loadtxt('demo.dat', dtype=dtype)


line = linefit.Line2dErr(dat['x'], dat['y'],
                         sigmax=data['sx'], sigmay=dat['sy'],
                         rhoxy=dat['rxy'])


