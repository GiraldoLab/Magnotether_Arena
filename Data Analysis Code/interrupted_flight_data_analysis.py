from turtle import delay
import matplotlib.pyplot as plt
import csv
import pandas as pd
from dateutil import parser
import numpy as np
import scipy as sp
import os
import glob
from circstats import difference, wrapdiff
 

# Calculates mean heading angle of fly
def full_arctan(x,y):
    angle = np.arctan(y/x)+np.pi if x<0 else np.arctan(y/x)
    return angle if angle <= np.pi else angle -2*np.pi  

def angle_avg(data):
    return full_arctan(np.cos(data*np.pi/180).sum(),np.sin(data*np.pi/180).sum())*180/np.pi  
def circmean(alpha, axis =None):   ###This is when averaging angles in radians
    mean_angle = np.arctan2(np.mean(np.sin(alpha), axis),np.mean(np.cos(alpha), axis))
    return mean_angle

#Calculates variance 
def angle_var(data):                                #Returns same value as 'circvar' in flight_arena_circular_analysis.py
    return 1-np.sqrt(np.sin(data*np.pi/180).sum()**2 + np.cos(data*np.pi/180).sum()**2)/len(data)

def angle_std(variance):
    return np.sqrt(-2*np.log(1-variance))
def yamartino_angle_std(variance):
    e = np.sqrt(1-(1-variance)**2)
    return np.arcsin(e)*(1+(2/np.sqrt(3)-1)*e**3)

def circdiff(alpha, beta):
    D = np.arctan2(np.sin(alpha*np.pi/180-beta*np.pi/180),np.cos(alpha*np.pi/180-beta*np.pi/180))
    return D

def sun_angle(data):
    if data == 19:
        return 135
    elif data == 57:
        return 45
    elif data == 93:
        return -45
    elif data == 129:
        return -135
    else:
        return 0

allFliesMeanAnglesListA = list()
allFliesRotatedMeanAnglesListA = list()
allFliesAngleVarsListA = list()
allFliesSunListA = list()

allFliesMeanAnglesListB = list()
allFliesRotatedMeanAnglesListB = list()
allFliesAngleVarsListB = list()
allFliesSunListB = list()

#AtrialsMeanHeadings = [[],[]]
#allFliesRotatedMeanAnglesListA = list()
#AtrialsHeadingVars = [[],[]]
#AtrialsSunPositions = [[],[]]

#BtrialsMeanHeadings = []
#allFliesRotatedMeanAnglesListB = list()
#BtrialsHeadingVars= []
#BtrialsSunPositions = []

path = '/home/giraldolab/catkin_ws/src/magno-test/nodes/data/For Data Analysis/Flight Behavior Paper data/HCS-different sun-interrupted'
#Sorts all the csv files in order in 'path' and sorts it to A trials and B trials
experiment_names = glob.glob(os.path.join(path, "*.csv"))
experiment_names.sort(reverse=False)

experiment_namesA = [f for f in experiment_names if f[-5:]=='A.csv']
experiment_namesA.sort(reverse=False)
#Sorts all fly A trials in numerical order - file names needs to start from fly01,fly02,.. when naming file
experiment_namesB = [f for f in experiment_names if f[-5:]=='B.csv']
experiment_namesB.sort(reverse=False)
#print(experiment_names)



experiments =[]
output= {}
time_delay = 0

for experiment_name in experiment_names:
    experiment = pd.read_csv(experiment_name)
    output[str(experiment_name)] = {}
    experiment['Image Time'] = experiment['Image Time'].apply(parser.parse)
    experiment_data = experiment.values
    sun_change_indexes = [0]+[i for i in range(1,len(experiment_data)) if experiment_data[i,3]!=experiment_data[i-1,3]]
    sun_periods = [experiment[sun_change_indexes[-1]:-1]]
    for sun_period in sun_periods:
        delayed_sun_period = sun_period.loc[[(frame_time - sun_period['Image Time'].iloc[0]).seconds>time_delay for frame_time in sun_period['Image Time']]]
        sun_position = sun_angle(delayed_sun_period['Sun Position'].iloc[0]) #to show where sun stimulus was during the experiment
        #print('sun_position:', sun_position)
        heading_angle = angle_avg(delayed_sun_period['Heading Angle']) #original: pos_angles(delayed_sun_period['Heading Angle'])

        heading_angle_var = angle_var(delayed_sun_period['Heading Angle'])#original: pos_angles(delayed_sun_period['Heading Angle'])

        heading_angle_std = angle_std(heading_angle_var)

        heading_angle_yamartino_std = yamartino_angle_std(heading_angle_var)

        rotated_sun_position = sun_position - sun_position

        output[str(experiment_name)][str(sun_period["Sun Position"].iloc[0])]= {
            "heading angle": heading_angle,
            "heading angle var": heading_angle_var,
            "heading angle std": heading_angle_std,
            "heading angle yamartino_std": heading_angle_yamartino_std,
        } 
        if experiment_name in experiment_namesA:     
            allFliesMeanAnglesListA.append(heading_angle)
            allFliesAngleVarsListA.append(heading_angle_var)
            allFliesSunListA.append(sun_position)

        elif experiment_name in experiment_namesB:
            allFliesMeanAnglesListB.append(heading_angle)
            allFliesAngleVarsListB.append(heading_angle_var)
            allFliesSunListB.append(sun_position)

allFliesMeanAnglesA = np.array(allFliesMeanAnglesListA)
allFliesAngleVarsA = np.array(allFliesAngleVarsListA)         
allFliesSunA = np.array(allFliesSunListA)


allFliesMeanAnglesB = np.array(allFliesMeanAnglesListB)
allFliesAngleVarsB = np.array(allFliesAngleVarsListB)
allFliesSunB = np.array(allFliesSunListB)

##apply cutt off filter of vecstrength<0.2 ###
allFliesAngleVarsACutOff = np.zeros (1)
allFliesAngleVarsBCutOff = np.zeros (1)
allFliesMeanAnglesACutOff = np.zeros (1)
allFliesMeanAnglesBCutOff = np.zeros (1)
allFliesSunACutOff = np.zeros (1)
allFliesSunBCutOff = np.zeros (1)  ##added 10/20 HP JL

for x in range (allFliesAngleVarsA.size):
    if allFliesAngleVarsA[x] > 0.8 or allFliesAngleVarsB[x] > 0.8:                                      ####MAX_VAR=0.8
        pass
    else:
        allFliesAngleVarsACutOff = np.append(allFliesAngleVarsACutOff, allFliesAngleVarsA[x])
        allFliesMeanAnglesACutOff = np.append(allFliesMeanAnglesACutOff, allFliesMeanAnglesA[x])
        allFliesSunACutOff = np.append(allFliesSunACutOff, allFliesSunA[x])

        allFliesAngleVarsBCutOff = np.append(allFliesAngleVarsBCutOff, allFliesAngleVarsB[x])
        allFliesMeanAnglesBCutOff = np.append(allFliesMeanAnglesBCutOff, allFliesMeanAnglesB[x])
        allFliesSunBCutOff = np.append(allFliesSunBCutOff, allFliesSunB[x])

allFliesAngleVarsACutOff = allFliesAngleVarsACutOff[1:]
allFliesMeanAnglesACutOff = allFliesMeanAnglesACutOff[1:]
print('allFliesMeanAnglesACutOff:', allFliesMeanAnglesACutOff)
allFliesSunACutOff = allFliesSunACutOff[1:]

allFliesAngleVarsBCutOff = allFliesAngleVarsBCutOff[1:]
allFliesMeanAnglesBCutOff = allFliesMeanAnglesBCutOff[1:]
print('allFliesMeanAnglesBCutOff:', allFliesMeanAnglesBCutOff)
allFliesSunBCutOff = allFliesSunBCutOff[1:]

# print('A_Vars_cutoff', allFliesAngleVarsACutOff)
# print('A_meanheading_cutoff', allFliesMeanAnglesACutOff)
# print('A_sun_cutoff', allFliesSunACutOff)

# print('B_Vars_cutoff', allFliesAngleVarsBCutOff)
# print('B_meanheading_cutoff', allFliesMeanAnglesBCutOff)
# print('B_sun_cutoff', allFliesSunBCutOff)

print('Atrialvecstrengthsavg:',1-sum(allFliesAngleVarsACutOff)/len(allFliesAngleVarsACutOff))
print('Btrialvecstrengthsavg:',1-sum(allFliesAngleVarsBCutOff)/len(allFliesAngleVarsBCutOff))
print('allFliesAngleVarsListA:',allFliesAngleVarsListA)
print('A_Vars_cutoff', allFliesAngleVarsACutOff)
print('allFliesAngleVarsListB:',allFliesAngleVarsListB)
print('B_Vars_cutoff', allFliesAngleVarsBCutOff)

allFliesVecStrengthCutOffA = 1-allFliesAngleVarsACutOff
allFliesVecStrengthCutOffB = 1-allFliesAngleVarsBCutOff

rotated_heading_anglesA = difference(allFliesMeanAnglesACutOff, allFliesSunACutOff, deg= True)

rotated_heading_anglesB = difference(allFliesMeanAnglesBCutOff, allFliesSunBCutOff, deg= True)

heading_difference= difference(rotated_heading_anglesA ,rotated_heading_anglesB, deg=True)
print('heading difference', heading_difference)

############## For polar plot ###############
fig, (ax1,ax2,ax3) = plt.subplots(1,3, subplot_kw = {'projection': 'polar'}, sharex = True, sharey=True)
r=1

for i in range(len(rotated_heading_anglesA)):
    ax1.plot((0,rotated_heading_anglesA[i]*np.pi/180.0),(0,1.0-allFliesAngleVarsACutOff[i]),'k', linewidth=2)
    ax1.set_title('A Trials Heading')
    ax2.plot((0,rotated_heading_anglesB[i]*np.pi/180.0),(0,1.0-allFliesAngleVarsBCutOff[i]),'k', linewidth=2)
    ax2.set_title('B Trials Heading')
    ax3.plot((0,heading_difference[i]*np.pi/180.0),(0,1.0),'k', linewidth=2)
    ax3.set_title('Heading Change')

for ax in (ax1,ax2,ax3):
    ax.set_theta_zero_location ("N")
    ax.set_theta_direction(-1)
    ax.set_rlim((0, 1.0))
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    circle = plt.Circle((0.0, 0.0), 1., transform=ax.transData._b, edgecolor='k', linewidth=2, facecolor= 'w', zorder=0)
    ax.add_artist(circle)
    ax.axis('off')
ax.set_title('HCS-different sun-interrupted', fontsize =18)




def LinearPlot(list_x, list_y, list_x_error, list_y_error, x_axis = 'list_x', y_axis ='list_y', title =None):
    fig = plt.figure(figsize=(5,10))
    fig.set_facecolor('w')
    ax = fig.add_subplot(1,1,1)
    for axis in ['left','bottom']:
                ax.spines[axis].set_linewidth(3.0)
                ax.spines['left'].set_position(('outward', 20))
                ax.spines['bottom'].set_position(('outward', 0))
    ax.xaxis.set_tick_params(width=3.0, length=8.0, direction = 'in')
    ax.yaxis.set_tick_params(width=3.0, length=8.0, direction = 'in') 
    ax.scatter(list_x, list_y, s=35, color='k', zorder=10)
    ax.scatter(list_x, list_y +360.0, s=35, color='k', zorder=10)
    ax.errorbar(x=list_x, y=list_y, xerr=list_x_error*36, yerr= list_y_error*36, fmt='none', linewidth =3, ecolor=[.6,.6,.6], capsize = 0, zorder=5)
    ax.errorbar(x=list_x, y=list_y +360.0, xerr=list_x_error*36 , yerr= list_y_error*36, fmt='none', linewidth =3, ecolor=[.6,.6,.6], capsize = 0, zorder=5)
    ax.plot([-180,180], [-180,180], color='k', zorder=1, linewidth=3, linestyle='solid')
    ax.plot([-180,180], [180,540], color='k', zorder=1, linewidth=3, linestyle='solid')

    ax.set_title('Continuous Flight Trials', fontsize =18)
    ax.set_xticks((-180, 0, 180)) #for larger fig

    ax.set_yticks((-180, 0, 180,  360,  540)) # for large fig

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_bounds(-180,540)

    ax.spines['bottom'].set_bounds(-180,180)

    ax.set_yticklabels(('-180', '0', '180',  '360',  '540'), color='k', fontsize=12)
    ax.set_xticklabels(('-180', '0', '180'), color='k', fontsize=12)
    ax.set_ylabel(y_axis, fontsize=10)
    ax.set_xlabel(x_axis, fontsize=10)

    ax.axis('equal')
    if title is not None:  # Add title if specified
        ax.set_title(title)
    #fig.tight_layout()
    fig.savefig(title, transparent=True, dpi=600)
LinearPlot(rotated_heading_anglesA, rotated_heading_anglesB, allFliesAngleVarsACutOff,allFliesAngleVarsACutOff,x_axis =' First Sun Headings', y_axis = 'Second Sun Headings', title = 'First_Third Sun Headings')



##############################
def BootstrapAnalysis(list_A, list_B, title=None, NUM_RESAMPLES=10000):
    observedDiffs = circdiff(list_B, list_A)
    observedDiffMean = np.mean(np.abs(observedDiffs))

    # Run with the same randomization
    resampledDiffMeans = np.zeros(NUM_RESAMPLES, dtype='float')
    for resampleInd in range(NUM_RESAMPLES):
        resampledB = np.random.permutation(list_B)
        resampledDiffs = circdiff(resampledB, list_A)
        resampledDiffMean = np.mean(np.abs(resampledDiffs))
        resampledDiffMeans[resampleInd] = resampledDiffMean

    pval = np.sum(resampledDiffMeans <= observedDiffMean)/float(NUM_RESAMPLES)
    pval = np.around(pval, decimals=3)

    observed_diff=observedDiffMean*180./np.pi
    observed_diff=np.around(observed_diff, decimals=3)

    bootstrap_mean=np.mean(resampledDiffMeans)*180./np.pi
    bootstrap_mean=np.around(bootstrap_mean, decimals=3)

    # Plotting using Peter's circ diff
    print ('pval= ', pval)
    print ('observed diff= ', observed_diff)
    print ('bootstrap mean= ', bootstrap_mean)
    fig = plt.figure(figsize=(4,2.25))
    fig.set_facecolor('w')
    ax = fig.add_subplot(1, 1, 1)
    for axis in ['left','bottom']:
                ax.spines[axis].set_linewidth(3.0)
    ax.xaxis.set_tick_params(width=3.0, length = 6.0, direction = 'in')
    ax.yaxis.set_tick_params(width=3.0, length = 6.0, direction = 'in')
    ax.hist(resampledDiffMeans*180./np.pi, bins=15, histtype='stepfilled', color = [0.7, 0.7, 0.7])
    ax.axvline(observedDiffMean*180./np.pi, ymin=0.078, ymax=0.88, color='r', linewidth=2)
    # ax.text((observedDiffMean+0.6)*180./np.pi, NUM_RESAMPLES/30., 'pval = '+str(pval), fontsize=14)
    # ax.text((observedDiffMean+0.6)*180./np.pi, NUM_RESAMPLES/15., 'ob diff='+str(observed_diff), fontsize=14)
    # ax.text((observedDiffMean+0.6)*180./np.pi, NUM_RESAMPLES/20., 'bs mean= '+str(bootstrap_mean), fontsize=14)
    ## following commands to make figure "pretty"
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.set_xlabel('Mean angle difference',  fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xlim((-15, 200))
    ax.set_ylim((-200, 2500))
    ax.spines['bottom'].set_bounds(0, 180)
    ax.spines['left'].set_bounds(0, 2500)
    ax.set_xticks((0, 90, 180))
    ax.set_yticks((0, 2500))
    ax.set_yticklabels((0, 2500), fontsize=14)
    ax.set_xticklabels((0, 90, 180), fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title is not None:  # Add title if specified
        ax.set_title(title)
    fig.savefig(title, transparent=True, dpi=600)
    
# Call the function with your data
BootstrapAnalysis(rotated_heading_anglesA, rotated_heading_anglesB, 'First-Second Sun Bootstrap Results')


#% Plots Sea Urchin and runs Rayleigh Test
from circstats import confmean
from astropy.stats import rayleightest
from circstats import confmean
from collections import Counter

def sea_urchin_with_stacked_symbols(circmeans, vecstrengths, bin_num, hist_start,hist_spacing, figname):
    #fig = plt.figure(figsize = (10,10))
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection = 'polar')
    #ax.scatter(circmeans, vecstrengths, color = 'black', zorder=20) #for scatter plot
    ax.plot(circmeans, vecstrengths, color = 'black', linewidth = 2, zorder=20) #urchin
    circmeans=np.array(circmeans)
    pop_mean = (circmean(circmeans[0,:]))
    conf_in_95 = confmean(circmeans[0,:])
    CI_arc_r = np.ones(10000)
    print ('pop_mean = ', pop_mean)
    print ( 'conf_in_95 = ', conf_in_95)
    ax.plot((0, pop_mean), (0, 1), linewidth=2, color='r', zorder=22) # plot the circular mean
    #CI_arc = np.linspace((pop_mean-conf_in_95), (pop_mean + conf_in_95), num=10000)
    CI_arc = np.linspace(pop_mean-conf_in_95, pop_mean+conf_in_95, num=10000) % (2*np.pi)

    bins = np.linspace(0, 2*np.pi, bin_num+1, endpoint = True)
    digitized = np.digitize(circmeans[0,:], bins)
    z = Counter(digitized)
    ax.grid(False)
    circle = plt.Circle((0.0, 0.0), 1., transform=ax.transData._b, edgecolor=([0.9, 0.9, 0.9]), facecolor= ([0.9, 0.9, 0.9]), zorder=10)
    ax.plot(CI_arc, CI_arc_r, color= 'r', linewidth=2, zorder=50) #plot the umbrella
    ax.scatter(0, 0, s=75, color='r', marker= '+', linewidth = 2, zorder=25)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_zero_location ("N")
    ax.set_theta_direction(-1)
    ax.add_artist(circle)
    ax.spines['polar'].set_visible(False) 
    for bin_index, angle in enumerate(bins):
        #print angle, z[bin_index]
        count = z[bin_index]
        bin_spacing = 2*np.pi/bin_num
        bin_center = angle - (bin_spacing/2)  
        if count >0:
            hist_r_pos= np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True)
            #print ' hist_r_pos ', hist_r_pos
            #print 'diff', np.diff(hist_r_pos)
            ax.scatter([bin_center]*count, np.linspace(hist_start, hist_start+(hist_spacing*(count-1)), count, endpoint = True),s=95, marker = '.', color = 'black', zorder=20)
            ax.set_theta_offset(np.pi/2)
    ax.axis('off') #original 'off'
    ax.grid(False)
    ax.set_rmax(1.5)
    plt.tight_layout()
    fig.savefig(figname, transparent=True, dpi=600)

def PlotSeaUrchin(headings, vec_strengths, figname):
   
    #convert angles and angles_to_plot to radians and positive
    allFliesMeanAnglesCutOffRad= np.deg2rad(headings)
    a= np.array([np.mod(i+(2*np.pi), 2*np.pi) for i in allFliesMeanAnglesCutOffRad])
    r_A= vec_strengths
    AToPlot = np.concatenate((a[:, np.newaxis], np.zeros_like(a)[:, np.newaxis]), axis=1).T
    RAToPlot = np.concatenate((r_A[:, np.newaxis], np.zeros_like(r_A)[:, np.newaxis]), axis=1).T

    list_of_positive_angles = [(q +2*np.pi)%(2*np.pi) for q in allFliesMeanAnglesCutOffRad]
    list_of_positive_angles_ToPlot = [(q +2*np.pi)%(2*np.pi) for q in AToPlot]
    
    # Run Rayleigh test
    rayleigh_result = rayleightest(allFliesMeanAnglesCutOffRad) 
    print(f'Rayleigh Test Results: {rayleigh_result}')
        
    # plot
    sea_urchin_with_stacked_symbols(circmeans= list_of_positive_angles_ToPlot, vecstrengths= RAToPlot, bin_num=90, hist_start=1.1, hist_spacing=0.09, figname= figname)

PlotSeaUrchin(rotated_heading_anglesA, allFliesVecStrengthCutOffA, 'First Sun Polar plot')
PlotSeaUrchin(rotated_heading_anglesB, allFliesVecStrengthCutOffB, 'Third Sun Polar plot')
PlotSeaUrchin(heading_difference, np.ones_like(heading_difference), 'Heading Difference polarplot')


plt.show() 