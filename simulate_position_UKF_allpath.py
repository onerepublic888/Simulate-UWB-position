import numpy as np
from scipy.optimize import lsq_linear, root, minimize, basinhopping, fsolve
import matplotlib.pyplot as plt
from sympy import diff, Symbol, sin, tan
from scipy import optimize
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise, Saver
from filterpy.kalman import KalmanFilter

def simple_Kalman_filter(x, p, Q, R, m): #x: 前一值,p: covariance, Q: system noise ,R: 量測noise,m: 量測值 ,k: k gain
    x = x               # predicted x, assumed static transfer
    p = p + Q           # predicted p
    k = p / (p + R)     # k factor
    x = x + k * (m - x) # updated x
    p = (1. - k) * p     # updated p
    return x, p, k

def lsq_method(distances_to_anchors, anchor_positions):
    distances_to_anchors, anchor_positions = np.array(distances_to_anchors), np.array(anchor_positions)
    if not np.all(distances_to_anchors):
        raise ValueError('Bad uwb connection. distances_to_anchors must never be zero. ' + str(distances_to_anchors))
    anchor_offset = anchor_positions[0]
    anchor_positions = anchor_positions[1:] - anchor_offset
    K = np.sum(np.square(anchor_positions), axis=1)   #ax=1 列加
    squared_distances_to_anchors = np.square(distances_to_anchors)
    squared_distances_to_anchors = (squared_distances_to_anchors - squared_distances_to_anchors[0])[1:]
    b = (K - squared_distances_to_anchors) / 2.
    res = lsq_linear(anchor_positions, b, lsmr_tol='auto', verbose=0)
    return res.x + anchor_offset

def costfun_method(distances_to_anchors, anchor_positions):
    anchor_positions = np.array(anchor_positions)
    tag_pos = lsq_method(distances_to_anchors, anchor_positions)
    anc_z_ls_mean = np.mean(np.array([i[2] for i in anchor_positions]) )  
    new_z = (np.array([i[2] for i in anchor_positions]) - anc_z_ls_mean).reshape(4, 1)
    new_anc_pos = np.concatenate((np.delete(anchor_positions, 2, axis = 1), new_z ), axis=1)
    new_disto_anc = np.sqrt(abs(distances_to_anchors[:]**2 - (tag_pos[0] - new_anc_pos[:,0])**2 - (tag_pos[1] - new_anc_pos[:,1])**2))
    new_z = new_z.reshape(4,)

    a = (np.sum(new_disto_anc[:]**2) - 3*np.sum(new_z[:]**2))/len(anchor_positions)
    b = (np.sum((new_disto_anc[:]**2) * (new_z[:])) - np.sum(new_z[:]**3))/len(anchor_positions)
    # print('a,b: ',a,b)
    cost = lambda z: np.sum(((z - new_z[:])**4 - 2*(((new_disto_anc[:])*(z - new_z[:]))**2 ) + new_disto_anc[:]**4))/len(anchor_positions) 

    function = lambda z: z**3 - a*z + b
    derivative = lambda z: 3*z**2 - a

    def newton(function, derivative, x0, tolerance, number_of_max_iterations=100):
        x1 = 0
        if (abs(x0-x1)<= tolerance and abs((x0-x1)/x0)<= tolerance): return x0
        k = 1
        while(k <= number_of_max_iterations):
            x1 = x0 - (function(x0)/derivative(x0))
            if (abs(x0-x1)<= tolerance and abs((x0-x1)/x0)<= tolerance): return x1
            x0 = x1
            k = k + 1
            if (k > number_of_max_iterations):  print("ERROR: Exceeded max number of iterations", a, b)
        return x1 

    newton_z_from_postive = newton(function, derivative, 100, 0.01)
    # newton_z_from_negative = newton(function, derivative, -100, 0.01)

    # def find_newton_global(newton_z_from_postive, newton_z_from_negative):
    #     if cost(newton_z_from_postive) < cost(newton_z_from_negative):
    #         return newton_z_from_postive
    #     elif cost(newton_z_from_negative) < cost(newton_z_from_postive):
    #         return newton_z_from_negative

    # newton_z = find_newton_global(newton_z_from_postive, newton_z_from_negative)
    # print('from_postive, from_negative: ', newton_z_from_postive, newton_z_from_negative)
    # print('The approximate value of Height is: ' +str(newton_z))

    newton_z = newton_z_from_postive
    
    ranges = (slice(0, 100, 1), )
    resbrute = optimize.brute(cost, ranges, full_output = True, finish = optimize.fmin)
    # print('resbrute: ', resbrute[0] + anc_z_ls_mean)

    new_tag_min = np.concatenate((np.delete(np.array(tag_pos), 2), [newton_z] + anc_z_ls_mean))
    return new_tag_min, newton_z + anc_z_ls_mean, a, b , float(resbrute[0]) + anc_z_ls_mean, anc_z_ls_mean

def cal_cdf(pos_error):
    hist, bin_edges = np.histogram(pos_error, bins=20)
    # last_bin_edges =  round(bin_edges[-3], 3) 
    cdf_value = np.cumsum(hist/sum(hist))
    return bin_edges, cdf_value, np.percentile(pos_error,95)

plot_ls, plot_ls_KF, plot_ls_UKF = [],[],[]
all_err_ls, all_err_KF_ls, all_err_UKF_ls = [],[], []
for tag_z in range(21):
    
    a_value, b_value = [],[]
    z_newton_ls, z_brute_ls , pos_error_ls, pos_predictions, pos_predictions_error_ls , pos_error_brute_ls= [],[],[],[],[],[]
    tag_pos_ls, raw_dis_to_anc = [], []
    PY_KF_z_error, PY_UKF_error = [],[]   

    x,y,z = 60,60,1 
    # tag_z = 20
    sigma = 0.1
    step_inr = 40
    anchor_pos_list = [[0,0,1.5*z], [x,0,1*z], [x,y,1*z], [0,y,1*z]]
    x_list, y_list = list(range(10, 50, 1)), list(range(10, 50, 1))
    x_list, y_list = list(range(10, 50, 1)), list(range(10, 50, 1))
    inv_x_list, inv_y_list = list(range(50, 10, -1)), list(range(50, 10, -1))
    z_list = (tag_z * np.ones((step_inr,), dtype=int)).tolist()

    path_1 = np.array([(10 * np.ones((step_inr,), dtype=int)).tolist(), y_list, z_list]).T
    path_2 = np.array([x_list, (50 * np.ones((step_inr,), dtype=int)).tolist(), z_list]).T
    path_3 = np.array([(50 * np.ones((step_inr,), dtype=int)).tolist(), inv_y_list, z_list]).T
    path_4 = np.array([inv_x_list, (10 * np.ones((step_inr,), dtype=int)).tolist(), z_list]).T
    All_path = np.vstack((path_1, path_2, path_3, path_4))
    for i, tag_pos in enumerate(All_path):
        tag = np.array(tag_pos)
        distances_to_anchors_raw = [np.linalg.norm(anchor_pos_list[0] - tag, axis=0), np.linalg.norm(anchor_pos_list[1] - tag, axis=0),\
                                        np.linalg.norm(anchor_pos_list[2] - tag, axis=0), np.linalg.norm(anchor_pos_list[3] - tag, axis=0)] 
        dis_noise = [np.random.normal(0, scale = sigma), np.random.normal(0, scale = sigma), 
                        np.random.normal(0, scale = sigma), np.random.normal(0, scale = sigma)]
        distances_to_anchors = np.sum([dis_noise, distances_to_anchors_raw], axis = 0)

        cost_tag_pos, z_newton , a_val, b_val, z_brute, anc_z_ls_mean = costfun_method(distances_to_anchors, anchor_pos_list)
        raw_dis_to_anc.append(distances_to_anchors)
        tag_pos_ls.append(cost_tag_pos)
        pos_error = np.linalg.norm(z_newton - tag[2])
        pos_error_brute = np.linalg.norm(z_brute - tag[2])
        a_value.append(round(a_val, 2))
        b_value.append(round(b_val, 2))
        z_newton_ls.append(round(z_newton,2))
        z_brute_ls.append(round(z_brute, 2))
        pos_error_ls.append(round(pos_error,2))
        pos_error_brute_ls.append(round(pos_error_brute,2))


    #------------------------------ Use filterpy KF-----------------------------
    f = KalmanFilter (dim_x=2, dim_z=1)
    f.F = np.array([[1.,1.],[0.,1.]])
    f.H = np.array([[1.,0.]])              
    f.P = np.array([[1.,    0.], [   0., 1.] ])
    f.R = np.array([[sigma**2]])    # uwb dis std **2

    saver_kf = Saver(f)
    for i in range(len(All_path)):
        if i == 0:
            # dt = dut_t[0]
            dt = 0.1
            f.x = np.array([z_newton_ls[i], 0])    #  position,velocity
            f.F = np.array([[1, dt], [0, 1]])
            f.predict()
            saver_kf.save()
            continue
        # f.Q = Q_discrete_white_noise(2, dt=dt,
        #  var=2e-5, block_size=2)
        dt = 0.1
        f.Q = Q_discrete_white_noise(dim=2, dt=1, var=1e-4)   
        f.update(z = z_newton_ls[i])
        f.predict(F = np.array([[1, dt], [0, 1]]))
        saver_kf.save()
        
    z_newton_ls_kf = np.array(saver_kf.x)[:,0]
    for m in z_newton_ls_kf:
        pos_predictions_error = np.linalg.norm(m - tag[2])
        PY_KF_z_error.append(round(pos_predictions_error, 2))

    #------------------------------ Use filterpy UKF-----------------------------
    def fx(x, dt):
        F = np.array([[1, dt, 0,  0, 0,  0],
                    [0,  1, 0,  0, 0,  0],
                    [0,  0, 1, dt, 0,  0],
                    [0,  0, 0,  1, 0,  0],
                    [0,  0, 0,  0, 1, dt],
                    [0,  0, 0,  0, 0,  1]])
        return np.dot(F, x)

    def hx(x):
        pos = np.array([x[0], x[2], x[4]])      # x[0], x[2], x[4] => x, y, z
        dist_to_anchor = np.linalg.norm(pos - np.array(anchor_pos_list), axis=1)
        return dist_to_anchor

    dt = 0.1
    points = MerweScaledSigmaPoints(n=6, alpha=0.01, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=4, dt=dt, fx=fx, hx=hx, points=points)
    ukf.R = (sigma**2) * np.eye(4)
    ukf.P = 1 * np.eye(6)

    saver_ukf = Saver(ukf)
    np.set_printoptions(suppress=True) 
    for i in range(len(All_path)):
        if i == 0:
            # dt = 0.1
            # ukf.x = np.concatenate([tag_pos_ls[i], [0, 0, 0]])
            ukf.x = np.array([tag_pos_ls[i][0], 0, tag_pos_ls[i][1], 0, tag_pos_ls[i][2], 0])
            # print(ukf.x)
            # kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
            ukf.predict(dt=dt)
            saver_ukf.save()
            continue
        # dt =  dut_t[i] - dut_t[i-1]
        
        
        ukf.Q = Q_discrete_white_noise(2, dt=dt, var=1e-6, block_size=3)
        # print(ukf.Q)
        ukf.predict(dt=dt)
        
        ukf.update(z = raw_dis_to_anc[i])
        saver_ukf.save()

    # print(np.around(np.array(saver.x), 2))
    tag_pos_UKF_z = np.around(np.array(saver_ukf.x)[:,4], 2).reshape(len(All_path),1)
    tag_pos_UKF_y = np.around(np.array(saver_ukf.x)[:,2], 2).reshape(len(All_path),1)
    tag_pos_UKF_x = np.around(np.array(saver_ukf.x)[:,0], 2).reshape(len(All_path),1) 
    # tag_pos_UKF = np.concatenate((np.concatenate((tag_pos_UKF_x, tag_pos_UKF_y), axis = 1), tag_pos_UKF_z), axis = 1)
    tag_pos_UKF = np.hstack((np.hstack((tag_pos_UKF_x, tag_pos_UKF_y)), tag_pos_UKF_z))
    # print('tag_pos_UKF: ',tag_pos_UKF)

    for m, tag in zip(tag_pos_UKF, All_path):
        print('m, tag: ', m, tag)
        pos_predictions_error = np.linalg.norm(m[2] - tag[2])
        # print('pos predict error: ', round(pos_predictions_error, 2))
        PY_UKF_error.append(round(pos_predictions_error, 2))


    #----------------------------------------------------------------------


    dict = {'GT_x':   All_path[:,0],
            'GT_y':   All_path[:,1],
            'GT_z':   All_path[:,2],
            "z_newton": z_newton_ls,  
            "pos_error": pos_error_ls,
            "z_newton_KF": z_newton_ls_kf,
            "pos_error_KF": PY_KF_z_error, 
            # 'A_value': a_value, 
            # 'B_value': b_value, 
            'z_brute_ls': z_brute_ls, 
            'pos_error_brute': pos_error_brute_ls,
            
            'PY_UKF_x':   tag_pos_UKF[:,0],
            'PY_UKF_y':   tag_pos_UKF[:,1],
            'PY_UKF_z':   tag_pos_UKF[:,2],
            
            'PY_UKF_error_Z':   PY_UKF_error,
            }
    df = pd.DataFrame(dict)
    # print(df)
    print('tag_z: ', tag_z)
    print('anc_z_ls_mean: ', anc_z_ls_mean)
    # print(df.loc[df.pos_error.idxmax()])

    # for i in range(len(z_newton_ls)):
    #     a = a_value[i]
    #     b = b_value[i]
    #     function = lambda z: z**3 - a*z + b
    #     z = np.array(np.arange(-6, 7, 0.1))
    #     plt.xlabel('z', fontsize=20)
    #     plt.ylabel('f(z)', fontsize=20)
    #     plt.axhline(y=0, color='k', lw = 1)
    #     plt.axvline(x=0, color='k', lw = 1)
    #     plt.title('Derivative of cost function', fontsize=20)
    #     if b<0:
    #         strb = str(b)
    #     elif b>0:
    #         strb = '+'+str(b)
    #     plt.legend(labels = ['z^3-'+str(a)+'z'+strb], loc = 'lower right' , fontsize=10)
    #     plt.grid()
    #     plt.plot(z, function(z), color='k')
        
    #     plt.savefig('plot'+str(i)+'.png', dpi=1000)
    #     plt.close() 
        
    # filter1 = df["pos_error"] > 2.5
    # df.where(filter1)
    # print(df.where(filter1))
    # print(df["B_value"])
    # print('z_newton_ls: ', np.array(z_newton_ls))
    # print('pos_error_ls: ', np.array(pos_error_ls))
    # print('z_newton_ls_KF: ', np.array(pos_predictions))
    # print('pos_error_ls: ', np.array(pos_predictions_error_ls))

    bin_edges, cdf_value, percentile_95 = cal_cdf(np.array(pos_error_ls))
    print('Z = ' + str(tag_z) + ', 95error_newtown: ' +'\t'+ str(round(percentile_95, 4)) + ' meters')
    plot_ls.append(np.array([tag_z, round(percentile_95, 3)]))
    all_err_ls.append(z_newton_ls)

    bin_edges_kal, cdf_value_kal, percentile_95_KF = cal_cdf(np.array(PY_KF_z_error))
    print('Z = ' + str(tag_z) + ', 95error_KF: ' +'\t' + str(round(percentile_95_KF, 4)) + ' meters')
    plot_ls_KF.append(np.array([tag_z, round(percentile_95_KF, 3)]))
    all_err_KF_ls.append(pos_predictions)

    bin_edges_brute, cdf_value_brute, percentile_95_brute = cal_cdf(np.array(pos_error_brute_ls))
    # print('Z = ' + str(tag_z) + ', 95error_brute: ' +'\t' + str(round(percentile_95_brute, 4)) + ' meters')

    bin_edges_pyKF, cdf_value_pyKF, percentile_95_UKF = cal_cdf(np.array(PY_UKF_error))
    print('Z = ' + str(tag_z) + ', 95error_UKF: ' +'\t' + str(round(percentile_95_UKF, 4)) + ' meters')
    plot_ls_UKF.append(np.array([tag_z, round(percentile_95_UKF, 3)]))
    all_err_UKF_ls.append(pos_predictions)
    print('-------------------------------')

np.savez('plot_ls_10.npz', plot_ls = plot_ls, all_err_ls = all_err_ls)
np.savez('plot_ls_KF_10.npz', plot_ls_KF = plot_ls_KF, all_err_KF_ls = all_err_KF_ls)
np.savez('plot_ls_UKF_10.npz', plot_ls_UKF = plot_ls_UKF, all_err_UKF_ls = all_err_UKF_ls)

plt_z_error_data_10 = np.load('plot_ls_10.npz')
plt_z_error_data_KF_10 = np.load('plot_ls_KF_10.npz')
plt_z_error_data_UKF_10 = np.load('plot_ls_UKF_10.npz')

plt_z_error_10 = plt_z_error_data_10['plot_ls']
plt_z_KF_10 = plt_z_error_data_KF_10['plot_ls_KF']
plt_z_UKF_10 = plt_z_error_data_UKF_10['plot_ls_UKF']

# print('plt_z_error_10: ', plt_z_error_10)
print('----------------plt_z_UKF_10: ', plt_z_UKF_10)
plt.figure(0)
plt.plot(range(21), plt_z_error_10[:,1], 'ko--',linewidth=3, markersize=8)
plt.plot(range(21), plt_z_KF_10[:,1], 'ks-',linewidth=3, markersize=8)
plt.plot(range(21), plt_z_UKF_10[:,1], 'bs-',linewidth=3, markersize=8)

plt.xlabel('Height [m]', fontsize=20)
plt.ylabel('CDF 95 percent Error [m]', fontsize=20)
plt.legend(labels = ['Z raw data', 'Z with KF','Z with UKF'], loc = 'upper right' , fontsize=20)

plt.xticks(np.arange(0,21,1), fontsize=20)
plt.yticks(np.arange(0,9.5,1), fontsize=20)

plt.xlim(0,20)
plt.ylim(0,9)
plt.grid(True)


plt.figure(1)
plt.boxplot(np.array(all_err_ls).T, positions=np.arange(0,21,1).tolist(), sym = "o")
plt.xticks(np.arange(0,21,1), fontsize=20)
plt.yticks(np.arange(-10,30.5,5), fontsize=20)
plt.xlabel('Tag Height [m]', fontsize=20)
plt.ylabel('Simulate result [m]', fontsize=20)
plt.xlim(-1,21)
plt.ylim(-10,30)
plt.title('Raw data', fontsize=20)
plt.grid(True)
plt.axis('equal')

# plt.figure(2)
# plt.boxplot(np.array(all_err_kal_ls).T, positions=np.arange(0,21,1).tolist(), sym = "o")
# plt.xticks(np.arange(0,21,1), fontsize=20)
# plt.yticks(np.arange(-10,30.5,5), fontsize=20)
# plt.xlabel('Tag Height [m]', fontsize=20)
# plt.ylabel('Simulate result [m]', fontsize=20)
# plt.xlim(-1,21)
# plt.ylim(-10,30)
# plt.title('Raw data with KF', fontsize=20)
# plt.grid(True)
# plt.axis('equal')

plt.show()