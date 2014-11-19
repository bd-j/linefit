import sys
import numpy as np
import matplotlib.pyplot as pl
import triangle
import linefit
import emcee
from emcee.utils import sample_ball

dm = 24.47
maggies_to_cgs = 10**(-0.4*(2.406 + 5*np.log10(1528.0)))

pred = np.loadtxt('data/b15_predicted_kroupa.dat', skiprows =1)
spred =  np.loadtxt('data/b15_predicted.dat', skiprows =1)
#jpred = np.loadtxt('data/b15_mod_fuv_attenuated.dat')
ppred = np.loadtxt('../B15/sed/B15_predicted.dat',skiprows =1)


dat = 'PS_M31_MOS08-fd-int_nan.pixels.dat'

x = pred[:,0]#flux
x = maggies_to_cgs*10**(-0.4*(ppred[:,3]))
y = np.loadtxt('data/'+dat, skiprows =1)

snr = 10.
A = np.mean(np.sqrt(y))/snr
yerr = A * np.sqrt(y)

line = linefit.Line(x , y, ysigma = yerr)


def lnprob_prior(m, b, Pb, Vb, Yb):
    lnp_prior = -((m - 1.0)/0.05)**2 -((Pb)/0.01)**2
    condition = ((b < 0) or (b > 1e-13) or
                 (m < 0.5) or (m > 2.0) or
                 (Pb < 0) or (Pb > 0.5))
    if condition:
        lnp_prior = -np.inf
    return lnp_prior

def lnprobfn(theta):
    #print(theta)
    Pb, Vb, Yb = 0.0, 1.0, 0.0
    m, b = theta[0], theta[1]
    if len(theta) > 2:
        Pb, Yb, Vb  = theta[2], theta[3], theta[4]
    lnp_prior = lnprob_prior(m, b, Pb, Vb, Yb)
    if np.isfinite(lnp_prior):
        lnlike = line.lnlike(m, b, Pb = Pb, Vb = Vb, Yb = Yb)
        #print(lnlike)
        if np.isnan(lnlike):
            return -np.inf
        return lnlike #+ lnp_prior
    else:
        return -np.inf
    
    
#m, b, uncertainty scaling
p0 = np.array([1.0, 8e-16, 0.1, 1e-16, 1e-32])#, 1.0]
ndim = len(p0)
nwalkers, nburn, niter = 64, 500, 200
p0 = sample_ball( p0, p0*0.1, size =nwalkers)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn,
                                threads = 1, pool = None)
pos, prob, state = sampler.run_mcmc(p0, nburn)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, niter)

xx = 1e-17 * (np.arange(1000.0) + 1.0)
#pl.figure()
#pl.plot(line.Z[:,0], line.Z[:,1], '.')
#pl.plot(xx, xx * pos[0,0] + pos[0,1])
#pl.xscale('log')
#pl.yscale('log')

#pl.figure()
#pl.plot(sampler.flatchain[:,0], sampler.flatchain[:,1], '.')

theta = sampler.chain[0,-1,:]
fg = line.lnprob_fg(*theta[0:2])
bg = line.lnprob_bg(*theta[-2:])
odds = np.log((theta[2] * np.exp(bg))/((1-theta[2]) * np.exp(fg)))
odds = odds - np.mean(odds)
pl.figure()
pl.scatter(line.Z[:,0], line.Z[:,1], c = odds)
pl.xscale('log')
pl.yscale('log')
cb = pl.colorbar(orientation = 'vertical')
cb.set_label('ln P(outlier) + C')
pl.ylim(1e-16,1e-14)
pl.xlim(5e-17,1e-14)
pl.xlabel('F(FUV) predicted')
pl.ylabel('F(FUV) observed')
for i in range(0,64,2):
    pl.plot(xx, xx * pos[i,0] + pos[i,1], color = 'red', alpha = 0.1)
    
pl.show()
sys.exit()
fig = triangle.corner(sampler.flatchain, labels = [r'$m$',r'$b$',r'$P_b$', r'$Y_b$', r'$V_b$'])

pl.figure()
pl.plot(x, y,'mo', mew = 0, alpha = 0.5)
pl.title(d)
pl.ylabel(r'$F_\lambda$ (obs)')
pl.xlabel(r'$F_\lambda$ (pred, Ben)')
pl.xscale('log')
pl.yscale('log')
pl.plot(x, y-4e-16, 'co', mew = 0, alpha = 0.5, label= 'obs-4e16')
pl.plot(xx,xx, label = 'equality', color = 'green')
pl.legend(loc = 0)
pl.show()


