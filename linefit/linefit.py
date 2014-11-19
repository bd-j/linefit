#Implement some of Hogg's line fitting likelihood functions, including
# especially uncertainties in both dimension

import numpy as np
#import emcee

class Line(object):

    def __init__(self, x, y,
                 ysigma=1.0, xsigma=0.0, rhoxy=0.0):
        """
        Initialize the Line object with x and y values and optionally
        the gaussian uncertainty estimate and covariance of each
        point.
        """
        self.npoint = len(x)
        self.Z = np.array([x, y]).T
        S = self.build_covariance(ysigma=ysigma, xsigma=xsigma, rhoxy=rhoxy)

        
    def lnprob_bg(self, Yb, Vb, yscale=1.0, yjitter=0.0):

        delta = self.Z[:,1] - Yb
        var = Vb + (yscale*self.S[:,1,1] + yjitter**2)
        return -np.log(np.sqrt(2*np.pi*var)) -  delta**2 / (2 * var) 

    def lnprob_fg(self, delta, var):
        return -np.log(np.sqrt(2*np.pi*var)) - delta**2 / (2 * var) 

class Line2dErr(Line):

    def build_covariance(self, ysigma=1.0, xsigma=0.0, rhoxy=0.0):
        """
        Build covariance tensor
        """
        self.S = np.zeros([self.npoint, 2, 2])
        self.S[:,1,1] = ysigma**2
        if np.any(xsigma != 0.0):
            self.S[:,0,0] = xsigma**2
            self.S[:,0,1] = self.S[:,1,0] = rhoxy * xsigma * ysigma
                
    def lnlike(self, theta, b_perp, V=0.0,
               Pb=0, Yb=0, Vb=1.0,
               xscale=1., xjitter=0.,
               yscale=1., yjitter=0.):
        
        # Describe vector orthogonal to line
        v = np.array([-np.sin(theta), np.cos(theta)])
        # Project displacements and covariances along this vector.
        #print(self.Z.shape, v.shape, np.size(b_perp))
        delta = v.dot(self.Z.T) - b_perp
        S = self.effective_covar(yscale, yjitter, xscale, xjitter)
        Sigma = v.dot(S).dot(v)
        var = Sigma**2 + V
        
        pfg = self.lnprob_fg(delta, var)
        return pfg.sum()
        #like = (1 - Pb) * np.exp(pfg)
        #if Pb > 0:
        #    pbg = self.lnprob_bg(Yb, Vb, yscale=yscale, yjitter=yjitter)
        #    like += Pb * np.exp(pbg)
        #return np.log(like).sum()

    def effective_covar(self, yscale, yjitter, xscale, xjitter):
        S = self.S.copy()
        S[:,1,1] = (yscale * S[:,1,1]) + yjitter**2
        S[:,0,0] = (xscale * S[:,0,0]) + xjitter**2
        
        return S
    
    def coeffs(self, theta, b_perp):
        return np.tan(theta), b_perp/ np.cos(theta)

class Line1dErr(Line):

    def build_covariance(self, ysigma=1.0, **kwargs):
        """
        Build covariance matrix.
        """

        self.S = np.zeros([self.npoint, 2, 2])
        self.S[:,1,1] = ysigma**2

    def lnlike(self, m, b,
               yjitter=0, yscale=1,
               Pb=0, Yb=0, Vb=1.0):

        delta = self.Z[:,1] - (self.Z[:,0] * m + b)
        var = self.effective_covar(scale, jitter)         
        pfg = self.lnprob_fg(delta, var)

        like = (1 - Pb) * np.exp(pfg)
        if Pb > 0:
            pbg = self.lnprob_bg(Yb, Vb, yscale=yscale, yjitter=yjitter)
            like += Pb * np.exp(pbg)
        return np.log(like).sum()
 
    def effective_covar(self, yscale, yjitter):
        return (yscale * self.S[:,1,1]) + yjitter**2
