import numpy as np
from scipy.optimize import lsq_linear, root
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import combinations
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import csv, time
from math import cos, sin, radians
from operator import itemgetter
from matplotlib.patches import Rectangle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.animation as animation
from scipy import optimize
import pandas as pd
from localization_algorithm_threading import costfun_method, new_cost_method, lsq_method
                
def root_method(distances_to_anchors, anchor_positions):
    # print('In theroot of ls_pos_2: ', ls_pos_2)    #put in intial guess
    tag_pos = lsq_method(distances_to_anchors, anchor_positions)
    sol = root(lambda pos: np.linalg.norm(pos - anchor_positions, axis=1) - distances_to_anchors, tag_pos, method='lm')
    return sol.x

def simple_Kalman_filter(x, p, Q, R, m): #x: 前一值,p: covariance, Q: system noise ,R: 量測noise,m: 量測值 ,k: k gain
    x = x               # predicted x, assumed static transfer
    p = p + Q           # predicted p
    k = p / (p + R)     # k factor
    x = x + k * (m - x) # updated x
    p = (1. - k) * p     # updated p
    return x, p, k

def cal_cdf(pos_error):
    hist, bin_edges = np.histogram(pos_error, bins=20)
    # last_bin_edges =  round(bin_edges[-3], 3)  #cdf 95% error meters
    # last_bin_edges_list.append(last_bin_edges)
    cdf_value = np.cumsum(hist/sum(hist))
    # print('95percentile:  ',np.percentile(pos_error,95))
    return bin_edges, cdf_value, np.percentile(pos_error,95)

def calculate_error(ls_pos_array, All_path):
    pos_predictions,  kal_z_pos_error, pos_error, pos_k, only_kal_z_pos_predict = [] , [], [], [], []
    
    Q = np.array((0.001, 0.001, 0.001))    # system noise (variance)
    R = np.array((0.0001, 0.0001, 0.01))              # measurement noise (variance)
    last_x, last_p = np.zeros_like(R), np.ones_like(Q) *1.
    for m in ls_pos_array:
        last_x, last_p, k = simple_Kalman_filter(last_x, last_p, Q, R, m)
        # print('last_x: ', last_x)
        pos_predictions.append(last_x)
        pos_k.append(k)

    kal_x, kal_y, kal_z = [point[0] for point in pos_predictions], [point[1] for point in pos_predictions], [point[2] for point in pos_predictions] 
    dat_x,dat_y,dat_z = [point[0] for point in ls_pos_array],[point[1] for point in ls_pos_array],[point[2] for point in ls_pos_array]
    
    for i in range(len(pos_predictions)):
    #   print(pos_predictions[i][0],ls_pos_array[i][0], All_path[i][0])
        
        # kal_z_pos = np.array((ls_pos_array[i][0], ls_pos_array[i][1], pos_predictions[i][2]))
        kal_z_pos = np.array((pos_predictions[i][0], pos_predictions[i][1], pos_predictions[i][2]))
        only_kal_z_pos_predict.append(kal_z_pos)

    for i in range(len(pos_predictions)):
        error = np.linalg.norm(ls_pos_array[i] - All_path[i])
        pos_error.append(error)
        kal_z_error = np.linalg.norm(only_kal_z_pos_predict[i] - All_path[i])
        kal_z_pos_error.append(kal_z_error)
    
    

    return kal_z_pos_error, pos_error, kal_z, dat_x, dat_y, dat_z, kal_x, kal_y


tag_z = 2
x,y,z, z2 = 13,5.6,1, 1 
step_inr = 50

anchor_positions = [[0,0,z], [x,0,z], [x,y,z+0.5], [0,y,z]]
x_list, y_list = list(np.arange(2, 12, 0.2)), list(np.array([2.4]).repeat(step_inr))
z_list = (tag_z * np.ones((step_inr,), dtype=int)).tolist()

All_path = np.array([x_list, y_list, z_list]).T
# print('anchor_positions: ', anchor_positions)
# print('All_path: ', All_path)
last_bin_edges_list, tag_list, lsq_pos_error_ls, pos_error_ls, pos_error_2_ls = [], [], [], [],[]
dis_to_anchor_array, ls_pos_array_1, ls_pos_array_2, ls_pos_array_3 = [] ,[], [], []
flight_path = []

for i, tag_pos in enumerate(All_path):
    tag = np.array(tag_pos)
    flight_path.append(tag)
    num = 1                   #一點幾次            
    distances_to_anchors_raw = [np.linalg.norm(anchor_positions[0] - tag, axis=0), np.linalg.norm(anchor_positions[1] - tag, axis=0),\
                                np.linalg.norm(anchor_positions[2] - tag, axis=0), np.linalg.norm(anchor_positions[3] - tag, axis=0)] 
 
    for _ in range(num):
        sigma = 0.05
        dis_noise = [np.random.normal(0, sigma), np.random.normal(0, sigma), np.random.normal(0, sigma), np.random.normal(0, sigma)]
        distances_to_anchors = np.sum([dis_noise, distances_to_anchors_raw], axis = 0)
        dis_to_anchor_array.append(np.around(distances_to_anchors, 2))

        ls_pos_1 = lsq_method(distances_to_anchors, anchor_positions) 
        ls_pos_2 = new_cost_method(distances_to_anchors, anchor_positions)
        ls_pos_3 = costfun_method(distances_to_anchors, anchor_positions)

        ls_pos_array_1.append(np.around(ls_pos_1, 2))
        ls_pos_array_2.append(np.around(ls_pos_2, 2))
        ls_pos_array_3.append(np.around(ls_pos_3, 2)) 

        lsq_pos_error = np.around(np.linalg.norm(ls_pos_1[2] - tag[2]), 2) 
        lsq_pos_error_ls.append(lsq_pos_error)

        pos_error_2 = np.around(np.linalg.norm(ls_pos_2[2] - tag[2]), 2)
        pos_error_2_ls.append(pos_error_2)

        pos_error = np.around(np.linalg.norm(ls_pos_3[2] - tag[2]), 2)     
        pos_error_ls.append(pos_error)


        
# np.savez('UWB_distances_to_anchors.npz', dis_to_anchor_array = dis_to_anchor_array,\
#         ls_pos_array_1 = ls_pos_array_1,ls_pos_array_3 = ls_pos_array_3, All_path = All_path)

# data = np.load('UWB_distances_to_anchors.npz')
# ls_pos_array_1 = data['ls_pos_array_1']
# ls_pos_array_3 = data['ls_pos_array_3']

kal_z_pos_error_1, pos_error_1, kal_z_1, dat_x_1, dat_y_1, dat_z_1, kal_x_1, kal_y_1 = calculate_error(ls_pos_array_1, All_path)
kal_z_pos_error_2, pos_error_2, kal_z_2, dat_x_2, dat_y_2, dat_z_2, kal_x_2, kal_y_2 = calculate_error(ls_pos_array_2, All_path)
kal_z_pos_error_3, pos_error_3, kal_z_3, dat_x_3, dat_y_3, dat_z_3, kal_x_3, kal_y_3 = calculate_error(ls_pos_array_3, All_path)
# print('dat_z_3: ', dat_z_3)

bin_edges_1, cdf_value_1, cdf95_edges_1 = cal_cdf(lsq_pos_error_ls)
bin_edges_2, cdf_value_2, cdf95_edges_2 = cal_cdf(pos_error_2)
bin_edges_3, cdf_value_3, cdf95_edges_3 = cal_cdf(pos_error_ls)
print('cdf95 lsq: ', cdf95_edges_1)
print('cdf95 new_cost: ', cdf95_edges_2)
print('cdf95 cost: ', cdf95_edges_3)
bin_edges_kal_z1, cdf_value_kal_z1, cdf95_kal_z1 = cal_cdf(kal_z_pos_error_1)
# bin_edges_kal_z2, cdf_value_kal_z2, last_bin_edges_kal_z2 = cal_cdf(kal_z_pos_error_2)
bin_edges_kal_z3, cdf_value_kal_z3, cdf95_kal_z3 = cal_cdf(kal_z_pos_error_3)


dict_5m = {
        'flight_path': flight_path,
        'distance':  dis_to_anchor_array, 
        'tag_pos_5ms': ls_pos_array_3,
        'pos_error_ls': pos_error_ls,
        'new pos_error_ls': pos_error_2_ls,
        'lsq_tag_pos_5ms': ls_pos_array_1,
        'lsq_pos_error_ls': lsq_pos_error_ls,
        }
df_5m = pd.DataFrame(dict_5m)
pd.set_option('display.max_rows', df_5m.shape[0]+1)
print(df_5m)


# np.savez('sim_3m_err_sigma_10.npz', pos_error_ls = pos_error_ls, lsq_pos_error_ls = lsq_pos_error_ls)
np.savez('sim_5m_err_sigma_10.npz', pos_error_ls = pos_error_ls, lsq_pos_error_ls = lsq_pos_error_ls)


# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)

plt.figure(0)
ax = plt.gca()
ax.set_xlim([-1, 15])
ax.set_xlabel('X')
ax.set_ylim([-1, 10])
ax.set_ylabel('Y')
ax.plot(All_path[:,0],All_path[:,1], 'k', linewidth=1, label='Ground Truth')
for i in range(len(anchor_positions)):
    ax.scatter(anchor_positions[i][0],anchor_positions[i][1], color="k", s=100)
ax.plot(dat_x_1, dat_y_1, 'b', label='2D localization')
ax.legend()
ax.grid(True)
plt.axis('equal')

plt.figure(1)
ax2 = plt.gca()
ax2.plot(range(len(kal_z_3)), dat_z_3, 'ro-',linewidth=1, markersize=3)
ax2.plot(range(len(kal_z_3)), kal_z_3, 'ys-',linewidth=1, markersize=3)
ax2.set_title('Height ', fontsize=14)
ax2.set_xlabel('Step', fontsize=14)
ax2.set_ylabel('Z (m)', fontsize=16)
ax2.legend(labels = ['Z raw data','Z with KF'], loc = 'lower right' )
ax2.grid(True)


plt.figure(2)
ax3 = plt.gca()
ax3.plot(bin_edges_3[:-1], cdf_value_3,'ro-', linewidth=2, markersize=6)
ax3.plot(bin_edges_kal_z3[:-1], cdf_value_kal_z3,'ys-', linewidth=2, markersize=6)
# ax3.set_title('CDF of position error', fontsize=18)
ax3.set_xlabel('Position error(m)', fontsize=16)
ax3.set_ylabel('CDF', fontsize=18)
ax3.legend(labels = ['Cost function', 'Cost function with KF'], loc = 'lower right' )
ax3.grid(True) 


plt.show()