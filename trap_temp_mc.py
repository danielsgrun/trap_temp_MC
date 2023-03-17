## Code by: Daniel Schneider Grün ##
## Innsbruck, 2023 ##

from mc_temp_functions import *

#%% Taking and plotting the best fit

yyyy = '2023'
mm = '03'
dd = '16'
date = [yyyy, mm, dd]

bus_freq = 1.4e6
meas_num = 23

n_samples = int(1e3) # number of samples for MC simulation


P_tot = 21e-3 # total light power after the AOD
n_traps = 1
P = P_tot / n_traps # light power in each trap
w0 = 1.14*1e-6

temperatures = [10, 300, 1200]
waists = [w0, w0, 1]

# Temps = []
# TSigmas = []
# Params = []

# for i in range(6):
#   results = find_MC_fit(temperatures, waists, P_tot, n_traps,
#                            date, meas_num, bus_freq, n_samples,
#                            param='Loss583_1_AOM_freq', param_index=i,
#                            plot = False)
#   Params.append(results[0])
#   Temps.append(results[1])
#   TSigmas.append(results[2])

# Temps = np.array(Temps)
# TSigmas = np.array(TSigmas)
# Params = np.array(Params)

results = find_MC_fit(temperatures, waists, P_tot, n_traps,
                            date, meas_num, bus_freq, n_samples,
                            param='Loss583_1_AOM_atten', param_index=1,
                            plot = True,
                            beam_cut = 'None')

zR = pi*w0**2/lambd # Rayleigh length, m
U0 = P*alpha / (pi*c*e0*w0**2) # Trap depth
# U0 = 2.8e-3 * kB # # Rb87, 2.8mK trap depth
omega_perp = np.sqrt(4*U0/(m*w0**2)) # radial trap frequency, Hz
omega_par = np.sqrt(2*U0/(m*zR**2)) # longitudinal trap frequency, Hz
  
#%% Trap frequency via release-and-recapture

dt_in = 1e-8
t_fin = 1000e-6
dt_1 = 2*np.pi/omega_perp/4 * 1
# dt_1 = 2*np.pi/omega_par/4 * 1
dt_3 = 12e-6

times = np.arange(0,t_fin,dt_in)

T = 150e-6
n_samples = int(1e3)

frac,xt,yt,zt = mc_trap_dynamics(trapPower=P, 
                                 trapTemp=T,
                                 waist=w0,
                                 n_samples=n_samples, 
                                 times=times,
                                 delta_t1=dt_1,
                                 delta_t3=dt_3)

plt.figure()
for i in range(n_samples):
  # plt.plot(dt, np.sqrt(xt[:,i]**2+yt[:,i]**2), lw=1, color='k', alpha=0.03)
  plt.plot(1e6*dt, 1e6*xt[:,i], lw=1, color=[0,0,i/n_samples], alpha=0.03)
  # plt.scatter(dt, xt[:,i] , marker = 'None', lw=1, color=color[:,i], alpha=0.03)
# plt.ylim(-1.5*w0, 1.5*w0)
plt.xlim(1e6*min(dt), 1e6*max(dt))
plt.xlabel("$\Delta t_2$ ($\mu$s)")
plt.ylabel("x($\Delta t_2$) ($\mu m$)")

plt.figure()
for i in range(n_samples):
  # plt.plot(dt, np.sqrt(xt[:,i]**2+yt[:,i]**2), lw=1, color='k', alpha=0.03)
  plt.plot(1e6*dt, 1e6*zt[:,i], lw=1, color=[0,0,i/n_samples], alpha=0.03)
  # plt.scatter(dt, xt[:,i] , marker = 'None', lw=1, color=color[:,i], alpha=0.03)
# plt.ylim(-1.5*w0, 1.5*w0)
plt.xlim(1e6*min(dt), 1e6*max(dt))
plt.xlabel("$\Delta t_2$ ($\mu$s)")
plt.ylabel("z($\Delta t_2$) ($\mu m$)")
    

plt.figure()
plt.plot(1e6*dt[1:-1], frac[1:-1], color='k', alpha=0.8)
plt.xlabel("$\Delta t_2$ ($\mu$s)")
plt.ylabel("Normalized signal")
plt.ylim(-0.02,1.02)


#%% Varying T, re-estimating the radial freq. distributions
def gauss(x,amp,x0,sig):
    gaussian = amp*np.exp(-1/2 * (x-x0)**2/sig**2)
    return gaussian
Temp_list = 1e-6*np.array([60])
# Temp_list = 1e-6*np.linspace(20,300, 29)
# Temp_list = np.arange(0.02*U0/kB, 0.13*U0/kB+1e-9, 0.005*U0/kB)
n_samples = int(1e3)
dt_in = 1e-7
t_fin = 10000e-6
dt = np.arange(0,t_fin,dt_in)
dt_1 = 2*np.pi/omega_perp/4 * 0
# dt_1 = 2*np.pi/omega_par/4 * 1
dt_3 = 12e-6 * 0
fit_distributions = False
times = np.arange(0,t_fin,dt_in)
broadness_rd = []
central_freq_rd = []
broadness_ax = []
central_freq_ax = []
s = 0
for T in Temp_list:
  frac,xt,yt,zt = mc_trap_dynamics(trapPower=P, 
                                 trapTemp=T, 
                                 n_samples=n_samples, 
                                 times=times,
                                 delta_t1=dt_1,
                                 delta_t3=dt_3)
  
  # rt = np.sqrt(xt**2 + yt**2)
  freqs_rd, freqs_rd_dist = find_freqs_dist(xt, n_samples, dt_in)
  # freqs, freqs_dist = find_freqs_dist(rt, n_samples, dt_in)
  freqs_ax, freqs_ax_dist = find_freqs_dist(zt, n_samples, dt_in)
  
  freqs_ax = freqs_ax/1e3 # frequencies are now in kHz !!!
  freqs_rd = freqs_rd/1e3 # frequencies are now in kHz !!!
  
  peak_rd = np.where(freqs_rd_dist>0.98)[0][0]
  f_center_rd = freqs_rd[peak_rd]
  
  peak_ax = np.where(freqs_ax_dist>0.98)[0][0]
  f_center_ax = freqs_ax[peak_ax]
  
  if fit_distributions:
    fit_results_rd = curve_fit(gauss, freqs_rd, freqs_rd_dist, p0=[1,f_center_rd,10])[0]
    fit_results_ax = curve_fit(gauss, freqs_ax, freqs_ax_dist, p0=[1,f_center_ax,10])[0]
  
    broadness_rd.append(fit_results_rd[-1])
    central_freq_rd.append(fit_results_rd[-2])
  
    broadness_ax.append(fit_results_ax[-1])
    central_freq_ax.append(fit_results_ax[-2])
  else:
    pass  
  
  figure = plt.figure()
  plt.plot(freqs_rd, freqs_rd_dist+freqs_ax_dist, 'k-', lw=1.5)
  # plt.plot(freqs, freqs_rd_dist, 'k-', lw=1.5)
  # plt.title("Temperature = {0} $\mu$K".format(np.round(1e6*T)))
  plt.title("Temperature = {0}$\%$ $U_0/k_B$".format(np.round(kB*T/U0*100, decimals=1)))
  plt.xlabel("Frequency (kHz)")
  plt.ylabel("Distribution")
  plt.axvline(x=omega_perp/2/np.pi/1e3, color='r', alpha=0.7, ls='--')
  plt.axvline(x=omega_par/2/np.pi/1e3, color='r', alpha=0.7, ls='--')
  plt.savefig('fig_{0:04d}.png'.format(s))
  plt.close(figure)
  
  s += 1
  
broadness_ax = np.array(broadness_ax)
broadness_rd = np.array(broadness_rd)
central_freq_rd = np.array(central_freq_rd)
central_freq_ax = np.array(central_freq_ax)  
if fit_distributions:
    f,ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    ax1.plot(1e6*Temp_list, 2*broadness_rd/central_freq_rd, 'ko--', label="$\\nu_r$")
    ax1.set_xlabel("Temperature ($\mu$K)")
    ax1.set_ylabel("2$\sigma$ of $\omega_r$ dist. (kHz)")
    
    ax1.plot(1e6*Temp_list, 2*broadness_ax/central_freq_ax, 'ro--', label="$\\nu_z$")
    ax1.set_xlabel("Temperature ($\mu$K)")
    ax1.set_ylabel("2$\sigma$ of $\omega_z$ dist. (kHz)")
    plt.legend(loc=0)

#%% Inverse transform sampling

def random_num_fun(function):
    pass
    # vs = np.linspace(0,5*v_prob, 5*n_samples)
    # ys = maxwell_boltzmann(T,vs)
    # cdf = np.cumsum(ys) # cumulative distribution function, cdf
    # cdf = cdf/cdf.max()
    # mb_generator = interpolate.interp1d(cdf, vs)
    
    # v0 = mb_generator(np.random.rand(n_samples))
 
    # sin_theta, sin_phi = [1-2*np.random.rand(n_samples), 
    #                       1-2*np.random.rand(n_samples)]
    # cos_theta, cos_phi = [(-1)**np.random.randint(0,2,size=n_samples)*np.sqrt(1-sin_theta**2), 
    #                       (-1)**np.random.randint(0,2,size=n_samples)*np.sqrt(1-sin_phi**2)]
    
    # theta = np.pi*np.random.rand(n_samples)
    # phi = 2*np.pi*(1-2*np.random.rand(n_samples))
    
    # vz0 = v0*np.cos(theta)
    # vx0 = v0*np.sin(theta)*np.cos(phi)
    # vy0 = v0*np.sin(theta)*np.sin(phi)