#%% Physical constants and experimental conditions set up

## Code by: Daniel Schneider Grün ##
## Innsbruck, 2023 ##


from math import pi
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftfreq, fft
from scipy import interpolate
from scipy.optimize import curve_fit

from tqdm import tqdm 
from glob import glob

import pandas as pd

# from joblib import Parallel, delayed
import multiprocessing.dummy as mpd
import itertools as itt

pool = mpd.Pool()

amu = 1.66e-27 # Atomic mass unit --> kg
m = 166 * amu # Erbium166 isotope mass in kg
# m = 87 * amu # Rb87
kB = 1.38e-23 # Boltzmann constant, m^2 kg s^-2 K^-1
e0 = 8.854e-12 # electric permittivity, C^2 N^-1 m^-2
c = 299792458 # speed of light, m/s
g = 9.81 # gravitational acceleration, m/s^2

conversion = 0.1482e-24 * 1.113e-16 # conversion from a.u. to S.I.
alpha = 430 * conversion # polarizability
lambd = 488e-9 # laser wavelength, m

def parabola(x,a,b, x0):
  return a + b*(x-x0)**2

def maxwell_boltzmann(T, v):
  m_b = (m/(2*pi*kB*T))**1.5 * 4*pi * v**2 * np.exp(-m*v**2/(2*kB*T))
  return m_b

def get_unique_values(date, meas_num, param_name):
  '''
    Parameters
    ----------
    date : String array
        ['yyyy', 'mm', 'dd']
    meas_num : int
        Measurement number
    param_name : String
        Name of the parameter on the "x" axis

    Returns
    -------
    list
        List containing: [parameter values, averaged fluorescence, error]

  '''
  
  yyyy, mm, dd = date  
  directory = '\\\\treqs_camera.local\\d\\images\\'+yyyy+'\\'+mm+'\\'+dd+'\\'
  meas_name = yyyy+'-'+mm+'-'+dd+'_meas{0:04d}'.format(meas_num)
  info_loc = glob(directory + meas_name+'*')[0]
  info = pd.read_csv(info_loc)  
  
  data_param = info[param_name].to_numpy()  
  unique_param = info[param_name].unique()
  num_unique_param = len(unique_param)
  
  fluoresc_avg = np.zeros((len(unique_param)))
  fluoresc_err = np.zeros((len(unique_param)))
  
  for i in range(num_unique_param):
    loc = np.where(data_param == unique_param[i])[0]
    fluoresc_avg[i] = np.average(info['Cam4_Trap1_Sum'].to_numpy()[loc])
    fluoresc_err[i] = np.std(info['Cam4_Trap1_Sum'].to_numpy()[loc])/np.sqrt(len(loc))
    
  return [unique_param, fluoresc_avg, fluoresc_err]  

def load_data1d(date, meas_num, bus_freq):
  '''
    

    Parameters
    ----------
    date : String array
        Should be: ['yyyy', 'mm', 'dd']
    meas_num : integer
        Measurement number
    bus_freq : float
        Bus frequency used during the measurement (Hz !!!!!)

    Returns
    -------
    list
        Returns list containing: [time vector,
                                  averaged signal,
                                  uncertainty,
                                  standard error]

  '''
    
  # Loading the raw data
  
  yyyy, mm, dd = date
  
  directory = '\\\\treqs_camera.local\\d\\images\\'+yyyy+'\\'+mm+'\\'+dd+'\\'
  meas_name = yyyy+'-'+mm+'-'+dd+'_meas{0:04d}'.format(meas_num)
  info_loc = glob(directory + meas_name+'*')[0]
  df = pd.read_csv(info_loc)  
  
  # Manipulating the raw data
  times = df['ReleaseRecapture_release_time'].to_numpy()
  dt_data = np.unique(times).copy()
  dt_data = dt_data[:-1]

  natom = df['Cam4_Trap1_Sum']
  natom = natom.to_numpy()
  natom_avg_data = np.zeros_like(dt_data)
  error_data = np.zeros_like(dt_data)
  uncert_data = np.zeros_like(dt_data)

  for i in range(len(dt_data)):
    loc = np.where(times == dt_data[i])[0]
    natom_avg_data[i] = np.average(natom[loc])
    uncert_data[i] = np.std(natom[loc])
    error_data[i] = uncert_data[i]/np.sqrt(len(loc))  

  continuum = np.average([natom[i] for i in np.where(times==250)][0])

  dt_data = dt_data / bus_freq # converting from cycle times to us
  natom_avg_data = natom_avg_data - continuum
  uncert_data = uncert_data / natom_avg_data[0]
  error_data = error_data / natom_avg_data[0]
  loc_outliers = natom_avg_data > 1.2*natom_avg_data[0]
  [dt_data, natom_avg_data, uncert_data, error_data] = [np.delete(dt_data, loc_outliers),
                                                        np.delete(natom_avg_data, loc_outliers),
                                                        np.delete(uncert_data, loc_outliers),
                                                        np.delete(error_data, loc_outliers)]
  # natom_avg_data = natom_avg_data / np.max(natom_avg_data)
  natom_avg_data = natom_avg_data/natom_avg_data[0]
  
  return [dt_data, natom_avg_data, uncert_data, error_data]

def load_data2d(date, meas_num, bus_freq,
                param_name, param_index):
  '''
    

    Parameters
    ----------
    date : String array
        Should be: ['yyyy', 'mm', 'dd']
    meas_num : Integer
        Measurement number
    bus_freq : float
        Bus frequency used during the measurement (Hz !!!!!)
    param_name : String
        Name of the scanned parameter (other than time)
    param_index : integer > 0
        Index corresponding to the desired value of the scanned parameter.

    Returns
    -------
    list
        Returns list containing: [time vector,
                                  averaged signal,
                                  uncertainty,
                                  standard error,
                                  current value of the scanned parameter]

  '''
    
  yyyy, mm, dd = date
    
  # Loading the raw data...  
  directory = '\\\\treqs_camera.local\\d\\images\\'+yyyy+'\\'+mm+'\\'+dd+'\\'
  meas_name = yyyy+'-'+mm+'-'+dd+'_meas{0:04d}'.format(meas_num)
  info_loc = glob(directory + meas_name+'*')[0]
  df = pd.read_csv(info_loc)   
  
  #... for a given scanned parameter (other than time)  
  p = param_index
  params = np.unique(df[param_name].to_numpy())  
  locs = df.loc[df[param_name] == params[p]]
  
  # Manipulating the raw data
  times = locs['ReleaseRecapture_release_time'].to_numpy()
  dt_data = np.unique(times).copy()
  # dt_data = dt_data[:-1]
  
  natom = locs['Cam4_Trap1_Sum']
  # natom = locs['Cam4_Region3_Sum']
  natom = natom.to_numpy()
  natom_avg_data = np.zeros_like(dt_data)
  
  error_data = np.zeros_like(dt_data)
  uncert_data = np.zeros_like(dt_data)
  
  for i in range(len(dt_data)):
    loc = np.where(times == dt_data[i])[0]
    natom_avg_data[i] = np.average(natom[loc])
    uncert_data[i] = np.std(natom[loc])
    error_data[i] = uncert_data[i]/np.sqrt(len(loc))  

  # continuum = np.average([natom[i] for i in np.where(times==250)][0])
   
  loc_outliers = natom_avg_data > 1.2*natom_avg_data[0]
  [dt_data, natom_avg_data, uncert_data, error_data] = [np.delete(dt_data, loc_outliers),
                                                        np.delete(natom_avg_data, loc_outliers),
                                                        np.delete(uncert_data, loc_outliers),
                                                        np.delete(error_data, loc_outliers)]
  
  dt_data = dt_data / bus_freq # converting from cycle times to us
  # natom_avg_data = natom_avg_data - continuum
  uncert_data = uncert_data / natom_avg_data[0]
  error_data = error_data / natom_avg_data[0]
  # natom_avg_data = natom_avg_data / natom_avg_data[0]
  natom_avg_data = (natom_avg_data - np.min(natom_avg_data))/ (natom_avg_data[0]-np.min(natom_avg_data))
  # natom_avg_data = natom_avg_data / np.average(natom_avg_data[np.logical_and(natom_avg_data >= natom_avg_data[0],
                                                                # natom_avg_data < 1.2*natom_avg_data[0])])
  
  return [dt_data, natom_avg_data, uncert_data, error_data, params[param_index]]


def mc_temperatures(dt,
                    temperature, waist, n_samples, power, n_traps,
                    beam_cut = 'None'):

  # Defining some constants    
  conversion = 0.1482e-24 * 1.113e-16 # conversion from a.u. to S.I.
  alpha = 430 * conversion # polarizability  
  lambd = 488e-9 # laser wavelength, m
  
  P = power / n_traps

  T = temperature
  W0 = waist    
        
  u0 = P*alpha / (pi*c*e0*W0**2)

  if beam_cut != 'None':
    ZR = W0*beam_cut/np.sqrt(2)
    zR = W0*beam_cut/np.sqrt(2)
    
  else:  
    ZR = pi*W0**2/lambd
    zR = pi*W0**2/lambd

  omega_perp = np.sqrt(4*u0/(m*W0**2)) # radial trap frequency, Hz
  omega_par = np.sqrt(2*u0/(m*ZR**2)) # longitudinal trap frequency, Hz  
    
  dx_par = np.sqrt(kB*T/(m*omega_par**2))
  dx_perp = np.sqrt(kB*T/(m*omega_perp**2))
  dv = np.sqrt(kB*T/m)
    
  vz0 = np.random.normal(loc=0, scale=dv, size=n_samples)
  vy0 = np.random.normal(loc=0, scale=dv, size=n_samples)
  vx0 = np.random.normal(loc=0, scale=dv, size=n_samples)
    
    
  [x0, y0, z0] = np.array([np.random.normal(loc=0, scale=1*dx_perp, size=n_samples),
                           np.random.normal(loc=0, scale=1*dx_perp, size=n_samples),
                           np.random.normal(loc=0, scale=1*dx_par, size=n_samples)])
      
  ones = np.ones_like(np.outer(dt, x0))
  ones1d = np.ones_like(x0)
        
    
  [xt, yt, zt] = np.array([ones*x0 + np.outer(dt, vx0),
                           ones*y0 + np.outer(dt, vy0),
                           ones*z0 + np.outer(dt, vz0) - g/2 * np.outer(dt**2, ones1d)])
    
  [vxt, vyt, vzt] = np.array([ones*vx0,
                              ones*vy0,
                              ones*vz0 - g*np.outer(dt, ones1d)])
            
  kin_en = np.zeros((len(dt), n_samples))
  pot_en = np.zeros((len(dt), n_samples))
    
  for i in range(len(dt)):  
      kin_en[i] = m/2 * (vxt[i,:]**2 + vyt[i,:]**2 + vzt[i,:]**2)
      z_term = np.sqrt(1+(zt[i,:]/zR)**2)
      pot_en[i] = - u0/z_term**2 * np.exp(-2*(xt[i,:]**2 + yt[i,:]**2)/(W0**2*z_term**2)) + m*g*zt[i,:]
      # pot_en[i] = u0*(1 - 2*(xt[i]**2 + yt[i]**2)/w0**2 - 2*zt[i]**2/zR**2)
      
  energy = kin_en + pot_en
  frac = np.array([sum(energy[i,:] < 0)/n_samples for i in range(len(dt))])
  
  # print(omega_par/1e3)
      
  return frac  



def find_MC_fit(temperatures, waists,
                power, n_traps,
                date, meas_num, bus_freq, n_samples,
                param='None', param_index=0,
                plot = True,
                beam_cut = 'None'):
  '''
    

    Parameters
    ----------
    temperatures : Array
        Array containing: [Initial temperature, final temperature, # of temperatures]
    waists : Array
        Array containing: [Initial waist, final waist, # of waists]
    date : String array
        Array containing: ['yyyy', 'mm', 'dd']
    meas_num : integer
        Measurement number
    bus_freq : float
        Bus frequency used during the measurement (Hz !!!!!)
    param : String, optional
        Name of the scanned parameter (other than time). The default is 'None'
    param_index : integer >= 0, optional
        Index corresponding to the desired value of the scanned parameter.
        The default is 0
    plot : Bool
        Whether or not to plot the RnR + fit
        The default is True

    Returns
    -------
    Array containing [temperature, sigma_temperature]

  '''
  index0 = 0 # 0 for first one
  index1 = -1 # -1 for last one

  
  if param == "None":
    dt_data, natom_avg_data, uncert_data, error_data = load_data1d(date, meas_num, bus_freq)
  
  else:  
    dt_data, natom_avg_data, uncert_data, error_data, params = load_data2d(date, meas_num, bus_freq,
                                                                         param, param_index);

  if index1 == -1:
    dt = dt_data[index0:] # taking only up to 60 uK
    natom_avg = natom_avg_data[index0:] # ||
    error = error_data[index0:] # ||
  else:
    dt = dt_data[index0:index1] # taking only up to 60 uK
    natom_avg = natom_avg_data[index0:index1] # ||
    error = error_data[index0:index1] # ||  
  
  temp_0, temp_f, n_temp = temperatures
  temp_list = 1e-6*np.linspace(temp_0, temp_f, n_temp)
  
  w_0, w_f, n_waist = waists
  w0_list = np.linspace(w_0, w_f, n_waist)
  
  print("Calculating MC survival fractions")
  # print("Setting processes for MC survival fractions calculation")
  
  fracs = np.array([[mc_temperatures(dt,
                                      temp, waist, n_samples, power, n_traps,
                                      beam_cut = beam_cut)
                     for waist in w0_list]
                    for temp in tqdm(temp_list)])
  
  # def mc_Temps(args):
  #   temp, waist = args[0], args[1]  
  #   return mc_temperatures(dt, temp, waist, n_samples, P, n_traps)  
  
  # args = np.zeros((n_waist*n_temp, 2))  
  
  # s = 0  
  # for temp in temp_list:
  #   for waist in w0_list:
  #     args[s] = [temp, waist]
  #     s += 1

  # fracs = np.array([pool.map(mc_Temps, [args[i]]) 
  #                   for i in tqdm(range(len(args)))])
  
  # fracs = np.reshape(fracs, (n_temp, n_waist))
  
  
  print("Fitting MC survival fractions")
  
  devs = np.array([[abs(np.sum((fracs[i,j] - natom_avg)**2/error**2))
                    for j in range(len(w0_list))]
                   for i in tqdm(range(len(temp_list)))])

  loc_min = np.where(devs == np.min(devs))
  
  P = power/n_traps
  
  temp_min = np.array([temp_list[loc_min[0][0]]])
  w0_min = np.array([w0_list[loc_min[1][0]]])
  u0_min = P*alpha / (pi*c*e0*w0_min**2)

  delta_temp = temp_min/5
  n_temp_new = 100
  temp_list_new = np.linspace(temp_min-delta_temp, temp_min+1.5*delta_temp, n_temp_new)
  new_fracs = np.array([[mc_temperatures(dt,
                                         temp, waist, n_samples, power, n_traps,
                                         beam_cut = beam_cut)
                    for waist in w0_min]
                   for temp in temp_list_new])
  new_devs = np.array([[abs(np.sum((new_fracs[i,j] - natom_avg)**2/error**2))
                    for j in range(len(w0_min))]
                   for i in range(len(temp_list_new))])
  
  a0, deriv2, temp_min_new = curve_fit(parabola, 1e6*temp_list_new[:,0], new_devs[:,0],
                                   p0 = [np.min(devs),1e6*temp_min[0],1])[0]
  temp_min_new = 1e-6*temp_min_new
  
  sigma_temp = np.sqrt(2*(deriv2)**(-1))*1e-6
  
  if plot==True:

      dt_data = dt.copy()
      dt = np.linspace(dt[0], dt[-1], 500)
      frac = mc_temperatures(dt,
                             temp_min_new, w0_min, n_samples, P, n_traps,
                             beam_cut = beam_cut)
      frac_low = mc_temperatures(dt,
                                 temp_min_new-sigma_temp, w0_min, n_samples, P, n_traps,
                                 beam_cut = beam_cut)
      frac_high = mc_temperatures(dt,
                                  temp_min_new+sigma_temp, w0_min, n_samples, P, n_traps,
                                  beam_cut = beam_cut)
    
      plt.figure(figsize=(7,5))
      plt.fill_between(1e6*dt, y1=frac_low, y2=frac_high,
                       lw=2, color='r', alpha=0.7, 
               label="T = ({0:2.2f} $\pm$ {1:2.2f}) $\mu$K, $w_0$ = {2:2.0f} nm, $U_0$/$k_B$ = {3:2.2f} mK".format(1e6*temp_min_new,
                                                                                                                 1e6*sigma_temp,
                                                                                                                 1e9*w0_min[0],
                                                                                                                 1e3*u0_min[0]/kB))
      plt.plot(1e6*dt_data, natom_avg, color='k', lw=1, alpha=0.6)
      plt.errorbar(1e6*dt_data, natom_avg, yerr=error, capsize=2, color='k', fmt='o', alpha=0.7)
      plt.xlabel("$\Delta t$ $(\mu s)$")
      plt.ylabel("Recapture fraction")
      plt.legend(loc=0)
      if param != "None":
        plt.title("{0} = {1:2.2f}".format(param, params))
      # plt.title("Trap 4, $w_0$ free parameter")
      #plt.xticks(fontsize=18)
      #plt.yticks(fontsize=18)
      
    
      plt.figure(figsize=(7,5))
      plt.plot(1e6*temp_list_new[:,0], new_devs[:,0], label='Generated data', color='k')
      plt.plot(1e6*temp_list_new[:,0], parabola(1e6*temp_list_new[:,0], a0, deriv2, 1e6*temp_min_new), label='Parabolic fit', color='r')
      plt.legend(loc=0)
      plt.xlabel("Temperature ($\mu$K)")
      plt.ylabel("$\chi^2$")
  
  return [params, temp_min_new, sigma_temp]


def mc_trap_dynamics(trapPower,trapTemp, waist, n_samples,times, delta_t1=0, delta_t3=0):
  
  w0 = waist  
    
  dt_1 = delta_t1
  dt_3 = delta_t3
    
  dt_in = times[1]-times[0]  
  dt = times  
  
  T = trapTemp
  P = trapPower
  u0 = P*alpha / (pi*c*e0*w0**2)
  zR = pi*w0**2/lambd

  omega_perp = np.sqrt(4*u0/(m*w0**2)) # radial trap frequency, Hz
  omega_par = np.sqrt(2*u0/(m*zR**2)) # longitudinal trap frequency, Hz  
  
  dx_par = np.sqrt(kB*T/(m*omega_par**2))
  dx_perp = np.sqrt(kB*T/(m*omega_perp**2))
  dv = np.sqrt(kB*T/m)
    
  vz0 = np.random.normal(loc=0, scale=dv, size=n_samples)
  vy0 = np.random.normal(loc=0, scale=dv, size=n_samples)
  vx0 = np.random.normal(loc=0, scale=dv, size=n_samples)
    
    
  [x0, y0, z0] = np.array([np.random.normal(loc=0, scale=1*dx_perp, size=n_samples),
                           np.random.normal(loc=0, scale=1*dx_perp, size=n_samples),
                           np.random.normal(loc=0, scale=1*dx_par, size=n_samples)])
      
  ones = np.ones_like(np.outer(dt, x0))
  ones1d = np.ones_like(x0)
        
    
  [xt, yt, zt] = [np.zeros((len(dt), n_samples)),
                  np.zeros((len(dt), n_samples)),
                  np.zeros((len(dt), n_samples))]
    
  [vxt, vyt, vzt] = [np.zeros((len(dt), n_samples)),
                     np.zeros((len(dt), n_samples)),
                     np.zeros((len(dt), n_samples))]
  
  [vxt[0], vyt[0], vzt[0]] = [vx0, vy0, vz0 - g*dt_1]
  [xt[0], yt[0], zt[0]] = [x0 + vx0*dt_1, 
                           y0 + vy0*dt_1, 
                           z0 + vz0*dt_1 - 1/2*g*dt_1**2]
  
  kin_en = np.zeros((len(dt), n_samples))
  pot_en = np.zeros((len(dt), n_samples))
      
  kin_final = np.zeros_like(kin_en)
  pot_final = np.zeros_like(kin_en)


  kin_en[0] = m/2 * (vxt[0,:]**2 + vyt[0,:]**2 + vzt[0,:]**2)      
  z_term_old = np.sqrt(1+(zt[0,:]/zR)**2)
  pot_en[0] = - u0/z_term_old**2 * np.exp(-2*(xt[0,:]**2 + yt[0,:]**2)/(w0**2*z_term_old**2)) + m*g*zt[0,:]
        
  a_x_old = 4/m*xt[0]/(w0**2*z_term_old**2)*(pot_en[0]) 
  a_y_old = 4/m*yt[0]/(w0**2*z_term_old**2)*(pot_en[0])
  a_z_old = 2/m*zt[0]/(zR**4*w0**2*z_term_old**4)*(zR**2*(w0**2-2*(xt[0,:]**2+yt[0,:]**2)) + w0**2*zt[0,:]**2)*(pot_en[0])
      
  vxt[1] = vxt[0] + a_x_old*dt_in
  vyt[1] = vyt[0] + a_y_old*dt_in
  vzt[1] = vzt[0] + a_z_old*dt_in
      
  xt[1] = xt[0] + vxt[1]*dt_in
  yt[1] = yt[0] + vyt[1]*dt_in
  zt[1] = zt[0] + vzt[1]*dt_in
      
  kin_en[1] = m/2 * (vxt[1,:]**2 + vyt[1,:]**2 + vzt[1,:]**2)      
  z_term = np.sqrt(1+(zt[1,:]/zR)**2)
  pot_en[1] = - u0/z_term**2 * np.exp(-2*(xt[1,:]**2 + yt[1,:]**2)/(w0**2*z_term**2)) + m*g*zt[1,:]
      
  a_x = 4/m*xt[1]/(w0**2*z_term**2)*(pot_en[1]) 
  a_y = 4/m*yt[1]/(w0**2*z_term**2)*(pot_en[1])
  a_z = 2/m*zt[1]/(zR**4*w0**2*z_term**4)*(zR**2*(w0**2-2*(xt[1,:]**2+yt[1,:]**2)) + w0**2*zt[1,:]**2)*(pot_en[1])
  
  for i in tqdm(range(1,len(dt)-1)):
      
    xt[i+1] = xt[i-1] + vxt[i]*2*dt_in
    yt[i+1] = yt[i-1] + vyt[i]*2*dt_in
    zt[i+1] = zt[i-1] + vzt[i]*2*dt_in  
      
    kin_en[i] = m/2 * (vxt[i,:]**2 + vyt[i,:]**2 + vzt[i,:]**2)      
    z_term = np.sqrt(1+(zt[i,:]/zR)**2)
    pot_en[i] = - u0/z_term**2 * np.exp(-2*(xt[i,:]**2 + yt[i,:]**2)/(w0**2*z_term**2)) + m*g*zt[i,:]
    
    a_x = 4/m*xt[i]/(w0**2*z_term**2)*(pot_en[i]) 
    a_y = 4/m*yt[i]/(w0**2*z_term**2)*(pot_en[i])
    a_z = 2/m*zt[i]/(zR**4*w0**2*z_term**4)*(zR**2*(w0**2-2*(xt[i,:]**2+yt[i,:]**2)) + w0**2*zt[i,:]**2)*(pot_en[i])
    
    vxt[i+1] = vxt[i-1] + 2*a_x*dt_in
    vyt[i+1] = vyt[i-1] + 2*a_y*dt_in
    vzt[i+1] = vzt[i-1] + 2*a_z*dt_in
    
    vx_final, vy_final, vz_final = (vxt[i],
                                    vyt[i],
                                    vzt[i] - g*dt_3)
    
    x_final, y_final, z_final = (xt[i] + vx_final*dt_3,
                                 yt[i] + vy_final*dt_3,
                                 zt[i] + vz_final*dt_3 - g/2*dt_3**2)
    
    zterm_final = np.sqrt(1+(z_final/zR)**2)
    kin_final[i] = m/2 * (vxt[i,:]**2 + vyt[i,:]**2 + vzt[i,:]**2)
    pot_final[i] =  - u0/zterm_final**2 * np.exp(-2*(x_final**2 + y_final**2)/(w0**2*zterm_final**2)) + m*g*z_final
    
    
    
  energy_final = kin_final + pot_final
  # frac = np.array([sum(energy_final[i,:] < 0)/n_samples for i in range(len(dt))])
  frac = 0
  return [frac,xt,yt,zt]


def find_freqs_dist(array, n_samples, time_resolution):
    
  freqs_dist = 0
  for i in tqdm(range(n_samples)):
    freq_dist = fft(array[:,i])
    freqs_dist = freqs_dist + abs(freq_dist)
  freqs = fftfreq(len(array), d=time_resolution)
  freqs_dist = freqs_dist/np.max(abs(freqs_dist))
  pos_length = len(freqs)//2
  freqs_plot = freqs[:pos_length]
  freqs_dist_plot = freqs_dist[:pos_length]
  # max_pos = np.where(abs(freqs_plot) >= 1.3*omega_perp/2/np.pi)[0][0]
  max_pos = np.where(abs(freqs_plot) >= 80e3)[0][0]
  # min_pos = np.where(abs(freqs_plot) >= 100e3)[0][0]
  min_pos = 0
  
  freqs_new = freqs_plot[min_pos:max_pos+1]
  freqs_dist_new = freqs_dist_plot[min_pos:max_pos+1]
  
  freqs_dist_new /= np.max(freqs_dist_new)
#  return [freqs_plot[:max_pos+1], freqs_dist_plot[:max_pos+1]]
  return [freqs_new, freqs_dist_new]
