# coding=utf-8
'''
GUSED-CD model
@author: zhangga
@date: 20210506
@lab: State Key Laboratory of Hydroscience and Engineering, Tsinghua University
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.integrate import odeint
from scipy import interpolate

def calc_settling_velocity(D):
    D_star = (1.65 * 9.81 / nu / nu)**(1 / 3) * D
    w = (math.sqrt(25 + 1.2 * D_star**2) - 5)**1.5 * nu / D
    return w

def calc_NSE(X, obs, sim):
    # calcultate the NSE of observed v.s.simulated bed elevation in the backwater region
    obs1 = obs[X<3]
    sim1 = sim[X<3]
    return 1 - np.sum((obs1-sim1)**2) / np.sum((obs1-np.mean(obs1))**2)

def calc_width(idx, h): # calculate the width according to the site and water depth
    if (idx>flume_idx):
        width = 2 * (elev[idx]-elev_o[idx] + h)
    else:
        w_d = flume_width * (1 - idx / flume_idx)
        w_u = dam_width - idx / flume_idx * (dam_width - flume_width)
        width = w_d + (elev[idx]-elev_o[idx] + h) / dam_height * (w_u - w_d)
    return width

def calc_normal_depth(idx, S_f, Q):
    l = 0
    r = dam_height - (elev_o[idx] - elev[idx])
    iter = 0
    w2 = calc_width(idx, 0)
    while iter < 20:  # binary search algorithm
        h = (l + r) / 2
        w1 = calc_width(idx, h)
        area = (w1 + w2) * h / 2
        P = 2 * math.sqrt(2) * h + w2
        R = area / P
        V = 1/manning * R**(2/3) * S_f **(1/2)
        Q1 = area * V
        if Q1<Q:
            l = h
        else:
            r = h
        iter += 1
    return h

def calc_capacity():
    vol = 0
    for i in range(backwater_region_idx):
        w_1 = calc_width(i, 0)
        w_2 = calc_width(i, spillway_height-elev[i])
        area = (w_1 + w_2) * (spillway_height - elev[i]) / 2
        vol += area * d_x
    return vol

def backwater(y, x, dzdx, q_o, n):
    # backwater equation
    dydx = (dzdx - n**2 * q_o**2 / y**(10 / 3)) / (1 - q_o**2 / 9.81 / y**3)
    dydx *= -1
    return dydx

def sedi_trans(conc, x, q_o, w, rmax, f):
    # non-equilibrium sediment transport equation
    dydx = (-alpha * w * conc + rmax * f) / q_o
    return dydx

if __name__ == '__main__':
    grav = 9.81
    manning = 0.05 # manning coefficient
    rho = 1000 # water density
    Sigma = 1600 # wet density
    poro = 0.4 # porosity
    nu = 1e-06 # viscocity
    alpha = 0.05 # recovery saturation coefficient of the non-equilibrium sediment transportation

    Q = 0.001 # inflow discharge

    d_x = 0.05
    no_x = 61
    d_t = 60
    no_t = 12 * 18
    X = np.linspace(0, no_x * d_x, no_x+1)
    Omega_o = 0.008 # threshold of stream power
    c_os = [128, 102, 91, 95, 88, 93, 84, 91, 87, 92, 89, 86, 93, 85, 97, 93, 80, 93] # initial sediment concentration of different runs
    F = 0.1 # fraction of stream power to transport sediment
    slp = 0.1

    # parameters of check-dam
    spillway_height = 0.25
    dam_height = 0.35
    dam_width = 2.4
    dam_length = 0.9
    flume_width = 0.7
    flume_idx = int(dam_length / d_x)
    backwater_length = 3
    backwater_region_idx = int(backwater_length / d_x)

    normal_depth = (manning * Q*2 /math.sqrt(slp)) ** 0.375
    slp_sine = slp/math.sqrt(1. + slp*slp)
    # normal_depth = manning*Q/math.sqrt(slp_sine)
    # normal_depth = normal_depth ** 0.6
    Fr_no = Q / normal_depth / math.sqrt(grav * normal_depth) # Froude number

    # particle size distribution
    ds = np.array([0.000005, 0.000010, 0.000015, 0.000025, 0.000050, 0.000075, 0.000100])
    fi = np.array([0.1738, 0.0819, 0.1001, 0.1862, 0.2573, 0.1364, 0.0743])
    n_class = len(ds) # number of size classes

    depth = np.zeros(no_x + 1) # depth at each site
    elev = np.zeros(no_x + 1) # elevation at each site
    normal_depth_list = np.zeros(no_x + 1) # normal depth at each site
    elev_all = np.zeros([no_x + 1, 20]) # bed elevation at each site of each run


    q_o_list = np.zeros(no_x + 1) # unit discharge at each site
    q_o_all = np.zeros([no_x + 1, no_t + 1]) # unit discharge at each site at each time step

    vel = np.zeros(no_x + 1) # velocity
    r_max = np.zeros(no_x + 1) # stream power

    conc = np.zeros(n_class) # sediment concentration
    conc_o = np.zeros(n_class) # sediment concentration at last time step
    conc_local = np.zeros([no_x + 1, n_class]) # sediment concentration at each site
    conc_all = np.zeros([no_x + 1, no_t + 1, n_class]) # sediment concentration at each site at each time step of each class

    w = np.zeros(n_class) # settling velocity
    eta = np.zeros(n_class) # settling velocity


    for i in range(n_class):
        w[i] = calc_settling_velocity(ds[i])
    sol_old = None

    # read the measured data
    try: # read locally
        df_obs = pd.read_csv('Sediment_Profiles_of_check_dam.csv')
    except: # read from github
        print('Reading measured sediment profiles from Github')
        url = "https://raw.githubusercontent.com/THUzhangga/GUSED-CD/main/Sediment_Profiles_of_check_dam.csv"
        df_obs = pd.read_csv(url)
        df_obs.to_csv('Sediment_Profiles_of_check_dam.csv', index=False)
    z_obs = df_obs[f'N{0}'].values - 0.809
    x_obs = df_obs['x'].values
    f_obs = interpolate.interp1d(x_obs, z_obs)
    z_obs_new = f_obs(X)
    # set up the initial bed elevation
    ynew = f_obs(X)
    elev = ynew.copy()
    elev_o = elev.copy()
    elev_all[:, 0] = elev_o

    sedi_out = 0
    sedi_in = 0
    CI = []
    TE = []
    for k in range(1, no_t+1):
        N = (k-1) // 12 + 1  # number of runs
        c_o = c_os[N-1]
        sedi_in += c_o * Q * d_t
        print('Total time step:%d, run:%d'%(k, N))
        capacity = calc_capacity()
        # determine the water depth at the dam
        if (spillway_height - depth[0] - elev[0] > 0.001 and capacity > d_t * Q):# check-dam is not full   
            Q_out = 0
            vel[0] = 0
            # water depth at the dam uplifted
            # find the end of the backwater region
            if (depth[0] == 0):
                water_level_idx = 1
            for i in range(water_level_idx, no_x+1):
                vol = 0
                for j in range(i):
                    w_1 = calc_width(j, depth[j])
                    w_2 = calc_width(j, elev[i]+depth[i]-elev[j]-depth[j])
                    area = (w_1 + w_2) * (elev[i]+depth[i] - elev[j]-depth[j]) / 2
                    vol += area * d_x

                if vol > Q * d_t:
                    water_level_idx = min(i, backwater_region_idx)
                    depth[0] = elev[i]+depth[i] - elev[0]
                    break
        else:
            # if check-dam is full or capacity is too small, overflow occurs
            # check if all the inflow could overflow
            l = 0
            r = 0.1
            iter = 0
            while iter<10:
                h = (l + r) / 2
                Q_out = flume_width * 1.4 * h **1.5
                vol = 0
                for j in range(backwater_region_idx):
                    w_1 = calc_width(j, depth[j])
                    w_2 = calc_width(j, spillway_height+h-elev[j]-depth[j])
                    area = (w_1 + w_2) * (spillway_height+h - elev[j]-depth[j]) / 2
                    vol += area * d_x
                if vol + Q_out * d_t > Q * d_t:
                    r = h
                else:
                    l = h
                iter += 1
            Q_out = flume_width * 1.4 * h **1.5
            depth[0] = spillway_height - elev[0] + h
            q_o_list[0] = Q_out / calc_width(0, depth[0])
            vel[0] = Q_out / depth[0]
            
        for i in range(1, no_x + 1): # determine the unit discharge at each site
            y0 = depth[i-1]
            normal_depth = calc_normal_depth(i, max((elev[i]-elev[i-1])/d_x, 0.01), Q)
            normal_depth_list[i] = normal_depth
            depth[i] = depth[i-1] # in fact, depth[i] has not been solved, use the depth at the former site
            if (Q_out == 0): # check-dam is not full
                if (i>water_level_idx): # beyond the current backwater region
                    if (i>flume_idx): # in the branch gully
                        q_o = Q / max(calc_width(i, depth[i]), 2*normal_depth)
                    else: # in the main gully
                        q_o = Q / calc_width(i, depth[i])
                else: # in the current backwater region
                    # inner discharge is assumed to be linearly related to the water surface area to the check-dam
                    if (water_level_idx > flume_idx):  # if the backwater region reaches the branch gully
                        w_1 = calc_width(flume_idx, depth[flume_idx])
                        h_1 = (water_level_idx-flume_idx) * d_x
                        area_1 = w_1 * h_1 / 2
                        w_2 = calc_width(0, depth[0])
                        area_2 = (w_1 + w_2) * dam_length / 2
                        area = area_1 + area_2
                        if (i > flume_idx):
                            area_i = area - calc_width(i, depth[i]) * (water_level_idx-i) * d_x / 2
                        else:
                            w_i = calc_width(i, depth[i])
                            w_2 = calc_width(0, depth[0])
                            area_i = (w_i+w_2) * (i * d_x) / 2
                    else:
                        w_1 = calc_width(water_level_idx, depth[water_level_idx])
                        w_2 = calc_width(0, depth[0])
                        area = (w_1 + w_2) * (water_level_idx * d_x) / 2
                        w_i = calc_width(i, depth[i])
                        area_i = (w_i + w_2) * (i * d_x) / 2

                    q_o = Q * (area_i/area) / max(calc_width(i, depth[i]), 2*normal_depth)
            else:  # check-dam is full
                q_o = Q_out / max(calc_width(i, depth[i]), 2*normal_depth)

            q_o_list[i] = q_o
            n = manning
            x = [i * d_x, (i + 1) * d_x]
            x = [0.05, 0.1]
            dzdx = max((elev[i] - elev[i-1])/d_x, 0)
            sol = odeint(backwater, y0, x, args=(dzdx, q_o, n)) # solve the backwater equation

            if np.sum(np.isnan(sol))>0 or sol[1][0]>1: # NAN occurs
                print('i', i, elev[i], elev[i-1])
                print('Q_out', Q_out)
                print(q_o, y0, x, dzdx, 1-q_o**2 / 9.81 / y0**3)
                print('dydx', -(dzdx - n**2 * q_o**2 / y0**(10 / 3)) / (1 - q_o**2 / 9.81 / y0**3))
                print(sol, sol_old)
                raise(ValueError('Nan in depth'))
            # print(sol)
            depth[i] = max(sol[1][0], normal_depth) # update water depth
            vel[i] = q_o / depth[i]
            sol_old = sol
        q_o_all[:, k] = q_o_list
        for i in range(no_x + 1):
            q_o = q_o_list[i]
            tmp = -(i - 1) * d_x
            w1 = calc_width(i, depth[i])
            w2 = calc_width(i, 0)
            area = (w1 + w2) * depth[i] / 2
            P = 2 * math.sqrt(2) * depth[i] + w2
            R = area / P
            S_f = manning * 2 * (q_o * w1 / area) ** 2 / R ** (4 / 3)
            # S_f =  manning**2 * q_o**2 / depth[i]**3.333 # old manning equation
            Omega = rho * grav * S_f * q_o
            if (Omega>Omega_o):
                r_max[no_x - i] = F * Sigma * (Omega - Omega_o) / (Sigma -
                                                            rho) / depth[i] / grav
            else:
                r_max[no_x - i] = 0.0

        # now calculate sediment
        conc = fi * c_o
        conc_o = fi * c_o
        x1 = 0
        x2 = x1 + d_x
        tot_mass = 0.
        d_mass = 0.
        d_z = 0.
        tmp = -no_x*d_x
        conc_local[0] = conc

        for i in range(1, no_x + 1):
            j = no_x - i
            q_o = q_o_list[j]
            Rmax = r_max[i]
            eta = w * conc / np.sum(w * conc)
            x = [i*d_x, (i+1)*d_x]
            if q_o > 1e-06:
                sol_s = np.zeros(n_class)
                for nc in range(n_class):
                    sol = odeint(sedi_trans, conc[nc], x, args=(q_o, w[nc], Rmax, eta[nc]))
                    sol_s[nc] = sol[1][0]
                if np.sum(sol_s) < np.sum(conc_o):  # only when the deposition occurs
                    conc = sol_s  # update sediment concentration
            # calculate the sediment deposition in each unit
            if (k % 12 == 1): # set the sediment concentration at the first time step of each run
                conc_all[no_x - i, k] = conc
                conc_local[no_x - i] = conc
                d_mass = 0
            else: # calculate the sediment deposition in the unit
                conc_local[no_x-i] = conc
                conc_all[no_x-i, k] = conc
                d_mass = (np.sum(conc_o)-np.sum(conc)) * q_o
            d_z = d_mass*d_t/Sigma/(1. - poro)/d_x
            elev[no_x - i] += d_z
            conc_o = conc.copy() # must copy the value
        sedi_out += np.sum(conc_local[0]) * Q_out * d_t

        # after each 12 minutes, there was a 24-h break.
        # water stored were clear and all the sediments were deposited
        if (k>0 and k % (720/d_t) ==0):
            # horizontal water surface
            depth = spillway_height - elev
            for i in range(no_x+1):
                w1 = calc_width(i, depth[i])
                w2 = calc_width(i, 0)
                area = (w1 + w2) * depth[i] / 2
                sedi_area = np.sum(conc_local[i, :]) * area / Sigma /(1. - poro)
                l = 0
                r = 0.35
                iter = 0
                while iter < 10:  # binary search algorithm
                    h = (l+r)/2
                    w3 = calc_width(i, h)
                    area2 = (w3 + w2) * h / 2
                    if area2<sedi_area:
                        l = h
                    else:
                        r = h
                    iter += 1
                elev[i] += h

            draw_profile = True
            if draw_profile:
                plt.figure()

                z_obs = df_obs[f'N{N}'].values - 0.809
                x_obs = df_obs['x'].values
                f_obs = interpolate.interp1d(x_obs, z_obs)
                z_obs_new = f_obs(X)

                NSE = calc_NSE(X, obs=z_obs_new, sim=elev)
                plt.plot(X, elev_o, '-', label='initial profile')
                plt.plot(X, elev, 'o-', label='simulation of run %d, NSE=%.3f' % (N, NSE), mfc='None')

                plt.plot(X, elev_all[:, N-1], '--', label='last run %d' % (N-1))
                plt.plot(X, z_obs_new, 'k-', label='observation of run %d' % (N))

                plt.legend()
                # plt.savefig('fig/Obs_and_sim_sediment_prpfiles_Run%d.png' % (N), dpi=150)
                plt.show()
            elev_all[:, N] = elev
            inflow = Q * 12 * d_t
            capacity = calc_capacity()
            sedi_trap = sedi_in - sedi_out
            trap_efficiency = sedi_trap / sedi_in
            CI.append(capacity/inflow)
            TE.append(trap_efficiency)
            sedi_in = 0
            sedi_out = 0

            depth *= 0 # clear water
    np.save('profiles.npy', elev_all)
    np.save('X.npy', X)