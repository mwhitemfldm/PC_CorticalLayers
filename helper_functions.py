# import packages ----------------------------------------------
import sys
import os
import random
#sys.path.append('B:/code/cshearer/py/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cs_basefunctions as cms
from scipy.stats import zscore,ttest_1samp,sem
import inference_task as inf
import statsmodels.api as sm
import scipy.stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, LeavePOut
import sklearn.metrics as sm
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statannot
from scipy.stats import bootstrap
sample_rate, ca_sample_rate = cms.sample_rate, cms.ca_sample_rate

## color palettes  ----------------------------------------------

l_pal = {'L23':'#F83B07', 'L56':'#1E88E5', 'Difference': '#A4C639'}
l_cols = ['#F83B07','#1E88E5']

from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', label='L23', markerfacecolor=l_pal['L23'], markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='L56', markerfacecolor=l_pal['L56'], markersize=10)]

## mouse days ---------------------------------------------
# recording day relative to start of task

mhb99_days = {'210128':0,'210129':1,'210130':2,'210131':3,'210201':4,'210202':5,
                '210203':6, '210204':7,'210205':8,'210206':9,'210208':10,'210209':11,
                '210210':12,'210211':13,'210212':14,}

mhb100_days = {'210301':0,'210302':1,'210303':2,'210304':3,'210305':4,
                '210306':5,'210307':6,'210308':7,'210309':8,'210310':9,
                '210311':10,'210312':11,'210313':12,'210314':13,'210315':14,
                '210316':15,'210317':16}

mhb108_days = {'210505':0,'210506':1,'210507':2,'210508':3,'210509':4,'210510':5,
                '210511':6,'210512':7,'210513':8,'210514':9,'210515':10,'210516':11,
                '210517':12,'210518':13,'210519':14,'210520':15,'210521':16}

mhb124_days = {'221017':0,'221018':1,'221019':2, '221020':3,'221021':4,
                '221022':5,'221023':6,'221024':7,'221025':8,'221026':9,'221027':10,
                '221028':11,'221029':12,'221030':13, '221031':14, '221101':15,'221102':16}

mouse_rec_days = {'mhb99':mhb99_days,'mhb100':mhb100_days, 'mhb108':mhb108_days, 'mhb124':mhb124_days }

# functions  ----------------------------------------------

def assign_layer(x_pos, xpos_23, xpos_56):
    """ assign cell to layer given layer boundaries 
        Layer 2/3 is left and 5/6 is right of the interval defined by xpos_23, xpos_56 """
    if x_pos <= xpos_23:
        layer = 'L23'
    elif x_pos >= xpos_56:
        layer = 'L56'
    else:
        layer = 'unassigned'
    return layer

def sess_type_idxs(stages, ca_sess_idxs):
    """ assignes session indices for al, tt, and rl trials"""
    ca_fal = np.array(stages[stages['desen'] == 'f al'].index.to_list()) - 1
    ca_ftt = np.array(stages[stages['desen'] == 'f tt'].index.to_list()) - 1
    ca_frl = np.array(stages[stages['desen'] == 'f rl'].index.to_list()) - 1
    al_idx = np.intersect1d(ca_sess_idxs, ca_fal).tolist()
    tt_idx = np.intersect1d(ca_sess_idxs, ca_ftt).tolist()
    rl_idx = np.intersect1d(ca_sess_idxs, ca_frl).tolist()
    
    return {'al':al_idx , 'tt': tt_idx, 'rl': rl_idx}
  
    
# Calcium imaging trace processing  ----------------------------------------------

def get_cuepop(mouse_id, rec_day, cue_pulse_names, sess_type=False, correct_trials = 'correct', shift = False):
    
    """ Collects mean neural activity during each cue presentation"""
    
    for _ in [1]:

        folder = '../../merged/'+mouse_id+'-'+rec_day+'/'
        b,bsnm = cms.readfoldernm(folder)
        stages,units,trodes = cms.getRecDayInfo(b)
        ca_events,ca_pulses,all_sess_idxs, = cms.load_ca_data(bsnm,b,folder,datatype='traces')
           
        n_ca_cells = np.size(ca_events[0].filter(regex='C'),axis=1)
        
        if sess_type:
            ca_sess_idxs = sess_type_idxs(stages, all_sess_idxs)[sess_type]
        else:
            ca_sess_idxs = all_sess_idxs
            
        for sii, si in enumerate(all_sess_idxs):
            if rec_day in ['221017']:
                ca_pulses[si] = ca_pulses[si][::3, :]
            if len(ca_events[sii]) > len(ca_pulses[si]):
                print('segment length longer than pulse, length difference: ', len(ca_events[sii]) - len(ca_pulses[si]))
                if (len(ca_events[sii]) - len(ca_pulses[si])) < 5:
                    ca_events[sii] = ca_events[sii].iloc[:len(ca_pulses[si]),:].copy()
                else:
                    print('mismatch significant')
                    break
            ca_events[sii]['Time (eeg)']=ca_pulses[si][:len(ca_events[sii]),0]
            ca_events[sii].columns = ca_events[sii].columns.str.strip()
            
        all_to_ca = []
        for idx in ca_sess_idxs:
            all_to_ca.append(np.where(np.array(all_sess_idxs) == idx)[0].tolist()[0])

        ca_events = [ca_events[jj] for jj in all_to_ca]
        if len(ca_events) == 0:
            break
        
        cue_pulses = cms.loadPulse(b,cue_pulse_names,len(stages))
         
        # ca data
        ca_mats = []
        for si in range(len(ca_sess_idxs)):
            ca_events[si].replace(to_replace=' nan',value=np.nan,inplace=True) # replace nulls with nans
            ca_mats.append((ca_events[si].filter(regex='C').to_numpy(dtype=float)).T) # filter for cell cols

        ca_mat_all = np.hstack(ca_mats) # no. cells x time steps
        zca = zscore(ca_mat_all,axis=1,nan_policy='omit') # z score across trials for each cell (across cols for each row)
        zca= np.where(np.isnan(zca),0,zca)
        
        # get time points of cues 
        ca_cue_pulses_ = cms.pulse_to_ca_pulse(cue_pulses,ca_events,ca_sess_idxs)
        ca_cue_pulses = {}
        for cue in ca_cue_pulses_:
            ca_cue_pulses[cue]=[]
        sess_tot = 0
        for sii,si in enumerate(ca_sess_idxs):
            for cue in ca_cue_pulses:
                if np.size(ca_cue_pulses_[cue][sii])>0:
                    ca_cue_pulses[cue].append(ca_cue_pulses_[cue][sii]+sess_tot)
            #tracking[si]['time']+=sess_tot/sample_rate
            sess_tot+=np.size(ca_mats[sii],axis=1)
        for cue in ca_cue_pulses:
            ca_cue_pulses[cue] = np.vstack(ca_cue_pulses[cue]) # combine across sessions
        
        if correct_trials:
            # Behavioural information
            tracking = cms.load_tracking(b)
            dispenser = np.load(b+'.dispenser',allow_pickle=True)

            # correct trial indices
            speed_scores = [[],[]]
            for cuei,cue in enumerate(cue_pulse_names):
                for si in range(len(stages)):
                    if si in ca_sess_idxs:
                        speed_score = np.zeros(np.size(cue_pulses[cue][si],axis=0))
                        for triali in range(np.size(cue_pulses[cue][si],axis=0)):
                            smpsa =  np.flatnonzero(tracking[si]['time']*sample_rate>cue_pulses[cue][si][triali,1])
                            smpsb = np.flatnonzero(tracking[si]['time']*sample_rate<cue_pulses[cue][si][triali,1]+10*sample_rate)
                            smps = np.intersect1d(smpsa,smpsb)
                            speed_score[triali] = np.sum((tracking[si]['speed'][smps]<1)&(dispenser['near_disp'][si][smps]))
                        speed_scores[cuei].append(speed_score)
            s_scores= np.hstack(speed_scores[0])
            w_scores = np.hstack(speed_scores[1])
            correct = {cue_pulse_names[0]: s_scores>0, cue_pulse_names[1]: w_scores==0}
            incorrect = {cue_pulse_names[0]: s_scores==0, cue_pulse_names[1]: w_scores>0}

        # collect ca activities for each cue                                      b
        cue_pop = {}
        for cue in ca_cue_pulses:
            no_time = np.zeros((len(ca_cue_pulses[cue]),n_ca_cells))
            for triali, trial in enumerate(ca_cue_pulses[cue]):
                if shift==1:
                    no_time[triali] = np.mean(zca[:,trial[1]:trial[1]+(trial[1] - trial[0])],axis=1)
                elif shift==-1:
                    no_time[triali] = np.mean(zca[:,trial[0]-(trial[1] - trial[0]):trial[0]],axis=1)
                else:
                    no_time[triali] = np.mean(zca[:,trial[0]:trial[1]],axis=1)
            if correct_trials == 'correct':
                cue_pop[cue] = no_time[correct[cue],: ] # avg across time period, taking only correct trials
            elif correct_trials == 'incorrect':
                cue_pop[cue] = no_time[incorrect[cue],: ]
            else:
                cue_pop[cue] = no_time    
            
        return cue_pop
    
def get_layer_cuepop(mouse_id, rec_day, cue_pulse_names, sess_type, correct_trials = 'correct', shift = False):
    
     """ Collects mean neural activity during each cue presentation for L23 and L56"""
    
    # load layer coordinates
    layer_split_df = pd.read_csv('layer_x_coordinates.csv')

    for _ in [1]:

        folder = '../../merged/'+mouse_id+'-'+rec_day+'/'
        b,bsnm = cms.readfoldernm(folder)
        stages,units,trodes = cms.getRecDayInfo(b)
        ca_events,ca_pulses,all_sess_idxs, = cms.load_ca_data(bsnm,b,folder,datatype='traces')
        ca_sess_idxs = sess_type_idxs(stages, all_sess_idxs)[sess_type]
            
        for sii, si in enumerate(all_sess_idxs):
            if rec_day in ['221017']:
                ca_pulses[si] = ca_pulses[si][::3, :]
            if len(ca_events[sii]) > len(ca_pulses[si]):
                print('segment length longer than pulse, length difference: ', len(ca_events[sii]) - len(ca_pulses[si]))
                if (len(ca_events[sii]) - len(ca_pulses[si])) < 5:
                    ca_events[sii] = ca_events[sii].iloc[:len(ca_pulses[si]),:].copy()
                else:
                    print('mismatch significant')
                    break
            ca_events[sii]['Time (eeg)']=ca_pulses[si][:len(ca_events[sii]),0]
            ca_events[sii].columns = ca_events[sii].columns.str.strip()
            
        all_to_ca = []
        for idx in ca_sess_idxs:
            all_to_ca.append(np.where(np.array(all_sess_idxs) == idx)[0].tolist()[0])

        ca_events = [ca_events[jj] for jj in all_to_ca]
        if len(ca_events) == 0:
            break
        
        cue_pulses = cms.loadPulse(b,cue_pulse_names,len(stages))
        
        #read layer info
        layer_split_df[layer_split_df['day'] == rec_day]
        layer_split_df['day'] = layer_split_df['day'].apply(str) # convert day type
        x23 = int(layer_split_df[(layer_split_df['day'] == rec_day) & (layer_split_df['mouse'] == mouse_id)].layer23_x.copy())
        x56 = int(layer_split_df[(layer_split_df['day'] == rec_day) & (layer_split_df['mouse'] == mouse_id)].layer56_x.copy())
        # read cell coordinates
        pos_df = pd.read_csv(folder + mouse_id[1:] + '_' + rec_day + '_traces-props.csv')
        pos_df = pos_df[pos_df['Status'] == 'accepted']
        # assign layer to cell
        pos_df['Layer'] = pos_df['CentroidX'].apply(assign_layer, args = (x23, x56))
        L23_cell_names = pos_df[pos_df['Layer'] == 'L23'].Name
        L56_cell_names = pos_df[pos_df['Layer'] == 'L56'].Name
        L_cell_names = [L23_cell_names, L56_cell_names]

        cue_pops = []
        for layer in range(2):
            # ca data
            n_ca_cells = len(L_cell_names[layer])
            ca_mats = []
            for si in range(len(ca_sess_idxs)):
                ca_events[si].replace(to_replace=' nan',value=np.nan,inplace=True) # replace nulls with nans
                layer_ca_events = ca_events[si].loc[:, ca_events[si].columns.isin(L_cell_names[layer])]
                ca_mats.append(layer_ca_events.to_numpy(dtype=float).T) # filter for cell cols

            ca_mat_all = np.hstack(ca_mats) # no. cells x time steps
            zca = zscore(ca_mat_all,axis=1,nan_policy='omit') # z score across trials for each cell (across cols for each row)
            zca= np.where(np.isnan(zca),0,zca)

            # get time points of cues 
            ca_cue_pulses_ = cms.pulse_to_ca_pulse(cue_pulses,ca_events,ca_sess_idxs)
            ca_cue_pulses = {}
            for cue in ca_cue_pulses_:
                ca_cue_pulses[cue]=[]
            sess_tot = 0
            for sii,si in enumerate(ca_sess_idxs):
                for cue in ca_cue_pulses:
                    if np.size(ca_cue_pulses_[cue][sii])>0:
                        ca_cue_pulses[cue].append(ca_cue_pulses_[cue][sii]+sess_tot)
                #tracking[si]['time']+=sess_tot/sample_rate
                sess_tot+=np.size(ca_mats[sii],axis=1)
            for cue in ca_cue_pulses:
                ca_cue_pulses[cue] = np.vstack(ca_cue_pulses[cue]) # combine across sessions

            if correct_trials:
                # Behavioural information
                tracking = cms.load_tracking(b)
                dispenser = np.load(b+'.dispenser',allow_pickle=True)

                # correct trial indices
                speed_scores = [[],[]]
                for cuei,cue in enumerate(cue_pulse_names):
                    for si in range(len(stages)):
                        if si in ca_sess_idxs:
                            speed_score = np.zeros(np.size(cue_pulses[cue][si],axis=0))
                            for triali in range(np.size(cue_pulses[cue][si],axis=0)):
                                smpsa =  np.flatnonzero(tracking[si]['time']*sample_rate>cue_pulses[cue][si][triali,1])
                                smpsb = np.flatnonzero(tracking[si]['time']*sample_rate<cue_pulses[cue][si][triali,1]+10*sample_rate)
                                smps = np.intersect1d(smpsa,smpsb)
                                speed_score[triali] = np.sum((tracking[si]['speed'][smps]<1)&(dispenser['near_disp'][si][smps]))
                            speed_scores[cuei].append(speed_score)
                s_scores= np.hstack(speed_scores[0])
                w_scores = np.hstack(speed_scores[1])
                correct = {cue_pulse_names[0]: s_scores>0, cue_pulse_names[1]: w_scores==0}
                incorrect = {cue_pulse_names[0]: s_scores==0, cue_pulse_names[1]: w_scores>0}

            # collect ca activities for each cue                                      b
            cue_pop = {}
            for cue in ca_cue_pulses:
                no_time = np.zeros((len(ca_cue_pulses[cue]),n_ca_cells))
                for triali, trial in enumerate(ca_cue_pulses[cue]):
                    if shift==1:
                        no_time[triali] = np.mean(zca[:,trial[1]:trial[1]+(trial[1] - trial[0])],axis=1)
                    elif shift==-1:
                        no_time[triali] = np.mean(zca[:,trial[0]-(trial[1] - trial[0]):trial[0]],axis=1)
                    else:
                        no_time[triali] = np.mean(zca[:,trial[0]:trial[1]],axis=1)
                if correct_trials == 'correct':
                    cue_pop[cue] = no_time[correct[cue],: ] # avg across time period, taking only correct trials
                elif correct_trials == 'incorrect':
                    cue_pop[cue] = no_time[incorrect[cue],: ]
                else:
                    cue_pop[cue] = no_time
            cue_pops.append(cue_pop)
            
        return cue_pops
    
def get_layer_cuepop_full(mouse_id, rec_day, cue_pulse_names, sess_type, correct_trials = 'correct', shift = False):
    
    # load layer coordinates
    layer_split_df = pd.read_csv('layer_x_coordinates.csv')

    for _ in [1]:

        folder = '../../merged/'+mouse_id+'-'+rec_day+'/'
        b,bsnm = cms.readfoldernm(folder)
        stages,units,trodes = cms.getRecDayInfo(b)
        ca_events,ca_pulses,all_sess_idxs, = cms.load_ca_data(bsnm,b,folder,datatype='traces')
        ca_sess_idxs = sess_type_idxs(stages, all_sess_idxs)[sess_type]
            
        for sii, si in enumerate(all_sess_idxs):
            if rec_day in ['221017']:
                ca_pulses[si] = ca_pulses[si][::3, :]
            if len(ca_events[sii]) > len(ca_pulses[si]):
                print('segment length longer than pulse, length difference: ', len(ca_events[sii]) - len(ca_pulses[si]))
                if (len(ca_events[sii]) - len(ca_pulses[si])) < 5:
                    ca_events[sii] = ca_events[sii].iloc[:len(ca_pulses[si]),:].copy()
                else:
                    print('mismatch significant')
                    break
            ca_events[sii]['Time (eeg)']=ca_pulses[si][:len(ca_events[sii]),0]
            ca_events[sii].columns = ca_events[sii].columns.str.strip()
            
        all_to_ca = []
        for idx in ca_sess_idxs:
            all_to_ca.append(np.where(np.array(all_sess_idxs) == idx)[0].tolist()[0])

        ca_events = [ca_events[jj] for jj in all_to_ca]
        if len(ca_events) == 0:
            break
        
        cue_pulses = cms.loadPulse(b,cue_pulse_names,len(stages))
        
        #read layer info
        layer_split_df[layer_split_df['day'] == rec_day]
        layer_split_df['day'] = layer_split_df['day'].apply(str) # convert day type
        x23 = int(layer_split_df[(layer_split_df['day'] == rec_day) & (layer_split_df['mouse'] == mouse_id)].layer23_x.copy())
        x56 = int(layer_split_df[(layer_split_df['day'] == rec_day) & (layer_split_df['mouse'] == mouse_id)].layer56_x.copy())
        # read cell coordinates
        pos_df = pd.read_csv(folder + mouse_id[1:] + '_' + rec_day + '_traces-props.csv')
        pos_df = pos_df[pos_df['Status'] == 'accepted']
        # assign layer to cell
        pos_df['Layer'] = pos_df['CentroidX'].apply(assign_layer, args = (x23, x56))
        L23_cell_names = pos_df[pos_df['Layer'] == 'L23'].Name
        L56_cell_names = pos_df[pos_df['Layer'] == 'L56'].Name
        L_cell_names = [L23_cell_names, L56_cell_names]

        cue_pops = []
        for layer in range(2):
            # ca data
            n_ca_cells = len(L_cell_names[layer])
            ca_mats = []
            for si in range(len(ca_sess_idxs)):
                ca_events[si].replace(to_replace=' nan',value=np.nan,inplace=True) # replace nulls with nans
                layer_ca_events = ca_events[si].loc[:, ca_events[si].columns.isin(L_cell_names[layer])]
                ca_mats.append(layer_ca_events.to_numpy(dtype=float).T) # filter for cell cols

            ca_mat_all = np.hstack(ca_mats) # no. cells x time steps
            zca = zscore(ca_mat_all,axis=1,nan_policy='omit') # z score across trials for each cell (across cols for each row)
            zca= np.where(np.isnan(zca),0,zca)

            # get time points of cues 
            ca_cue_pulses_ = cms.pulse_to_ca_pulse(cue_pulses,ca_events,ca_sess_idxs)
            ca_cue_pulses = {}
            for cue in ca_cue_pulses_:
                ca_cue_pulses[cue]=[]
            sess_tot = 0
            for sii,si in enumerate(ca_sess_idxs):
                for cue in ca_cue_pulses:
                    if np.size(ca_cue_pulses_[cue][sii])>0:
                        ca_cue_pulses[cue].append(ca_cue_pulses_[cue][sii]+sess_tot)
                #tracking[si]['time']+=sess_tot/sample_rate
                sess_tot+=np.size(ca_mats[sii],axis=1)
            for cue in ca_cue_pulses:
                ca_cue_pulses[cue] = np.vstack(ca_cue_pulses[cue]) # combine across sessions

            if correct_trials:
                # Behavioural information
                tracking = cms.load_tracking(b)
                dispenser = np.load(b+'.dispenser',allow_pickle=True)

                # correct trial indices
                speed_scores = [[],[]]
                for cuei,cue in enumerate(cue_pulse_names):
                    for si in range(len(stages)):
                        if si in ca_sess_idxs:
                            speed_score = np.zeros(np.size(cue_pulses[cue][si],axis=0))
                            for triali in range(np.size(cue_pulses[cue][si],axis=0)):
                                smpsa =  np.flatnonzero(tracking[si]['time']*sample_rate>cue_pulses[cue][si][triali,1])
                                smpsb = np.flatnonzero(tracking[si]['time']*sample_rate<cue_pulses[cue][si][triali,1]+10*sample_rate)
                                smps = np.intersect1d(smpsa,smpsb)
                                speed_score[triali] = np.sum((tracking[si]['speed'][smps]<1)&(dispenser['near_disp'][si][smps]))
                            speed_scores[cuei].append(speed_score)
                s_scores= np.hstack(speed_scores[0])
                w_scores = np.hstack(speed_scores[1])
                correct = {cue_pulse_names[0]: s_scores>0, cue_pulse_names[1]: w_scores==0}
                incorrect = {cue_pulse_names[0]: s_scores==0, cue_pulse_names[1]: w_scores>0}
            
            #collect ca activities for each cue                                      b
            cue_pop = {}
            for cue in ca_cue_pulses:
                trial_length = np.min(ca_cue_pulses[cue][:,1] - ca_cue_pulses[cue][:,0])
                no_time = np.zeros((len(ca_cue_pulses[cue]),n_ca_cells, trial_length))
                for triali, trial in enumerate(ca_cue_pulses[cue]):
                    if shift==1:
                        no_time[triali] = zca[:,trial[1]:trial[1] +trial_length]
                    elif shift==-1:
                        no_time[triali] = zca[:,trial[0]-trial_length:trial[0]]
                    else:
                        no_time[triali] = zca[:,trial[0]:trial[0]+trial_length]

                if correct_trials == 'correct':
                    cue_pop[cue] = no_time[correct[cue],:,: ] # avg across time period, taking only correct trials
                elif correct_trials == 'incorrect':
                    cue_pop[cue] = no_time[incorrect[cue],:,: ]
                else:
                    cue_pop[cue] = no_time

            cue_pops.append(cue_pop)
            
        return cue_pops
  

def extract_XY(cue_pop):
    
    """ extracts features and labels form dictionary of cues and responses"""
    
    cue_pulse_names = list(cue_pop.keys())
    X=[]
    Y_string=[]
    Y=[]
    n_cues = len(cue_pulse_names)
    n_trials = np.zeros(n_cues,dtype=int)
    for cuei,cue in enumerate(cue_pop):
        X.append(cue_pop[cue])
        n_trials[cuei]=np.size(cue_pop[cue],axis=0)
        for triali in range(n_trials[cuei]):
            Y.append(cuei)
            Y_string.append(cue)

    X = np.vstack(X)
    Y = np.array(Y)
    
    return X, Y, Y_string
    
def extract_interval_XY(cue_pop, interval_length, interval_num):
    
    """ extracts features and labels form dictionary of cues and responses"""
    
    cue_pulse_names = list(cue_pop.keys())
    X=[]
    Y_string=[]
    Y=[]
    n_cues = len(cue_pulse_names)
    n_trials = np.zeros(n_cues,dtype=int)
    for cuei,cue in enumerate(cue_pop):
        X.append(cue_pop[cue][:,:,interval_num*interval_length : (interval_num + 1)*interval_length].mean(axis=2))
        n_trials[cuei]=np.size(cue_pop[cue],axis=0)
        for triali in range(n_trials[cuei]):
            Y.append(cuei)
            Y_string.append(cue)

    X = np.vstack(X)
    Y = np.array(Y)
    
    return X, Y, Y_string

## Tone active cell classifier -------------------------------------------------------------------------------------------

    
def on_off_cuepop(mouse_id, rec_day, cue_pulse_names, correct_trials, layer, sess_type, shift = False):

    # load layer coordinates
    layer_split_df = pd.read_csv('layer_x_coordinates.csv')

    for _ in [1]:

        folder = '../../merged/'+mouse_id+'-'+rec_day+'/'
        b,bsnm = cms.readfoldernm(folder)
        stages,units,trodes = cms.getRecDayInfo(b)
        ca_events,ca_pulses,all_sess_idxs, = cms.load_ca_data(bsnm,b,folder,datatype='traces')

        n_ca_cells = np.size(ca_events[0].filter(regex='C'),axis=1)
        ca_sess_idxs = sess_type_idxs(stages, all_sess_idxs)[sess_type]

        for sii, si in enumerate(all_sess_idxs):
            if rec_day in ['221017']:
                ca_pulses[si] = ca_pulses[si][::3, :]
            if len(ca_events[sii]) > len(ca_pulses[si]):
                print('segment length longer than pulse, length difference: ', len(ca_events[sii]) - len(ca_pulses[si]))
                if (len(ca_events[sii]) - len(ca_pulses[si])) < 5:
                    ca_events[sii] = ca_events[sii].iloc[:len(ca_pulses[si]),:].copy()
                else:
                    print('mismatch significant')
                    break
            ca_events[sii]['Time (eeg)']=ca_pulses[si][:len(ca_events[sii]),0]
            ca_events[sii].columns = ca_events[sii].columns.str.strip()

        all_to_ca = []
        for idx in ca_sess_idxs:
            all_to_ca.append(np.where(np.array(all_sess_idxs) == idx)[0].tolist()[0])

        ca_events = [ca_events[jj] for jj in all_to_ca]
        if len(ca_events) == 0:
            break

        cue_pulses = cms.loadPulse(b,cue_pulse_names,len(stages))
        
        if layer:
            #read layer info
            layer_split_df[layer_split_df['day'] == rec_day]
            layer_split_df['day'] = layer_split_df['day'].apply(str) # convert day type
            x23 = int(layer_split_df[(layer_split_df['day'] == rec_day) & (layer_split_df['mouse'] == mouse_id)].layer23_x.copy())
            x56 = int(layer_split_df[(layer_split_df['day'] == rec_day) & (layer_split_df['mouse'] == mouse_id)].layer56_x.copy())
            # read cell coordinates
            pos_df = pd.read_csv(folder + mouse_id[1:] + '_' + rec_day + '_traces-props.csv')
            pos_df = pos_df[pos_df['Status'] == 'accepted']
            # assign layer to cell
            pos_df['Layer'] = pos_df['CentroidX'].apply(assign_layer, args = (x23, x56))
            L_cell_names = pos_df[pos_df['Layer'] == layer].Name
            n_ca_cells = len(L_cell_names)

            ca_mats = []
            for si in range(len(ca_sess_idxs)):
                ca_events[si].replace(to_replace=' nan',value=np.nan,inplace=True) # replace nulls with nans
                layer_ca_events = ca_events[si].loc[:, ca_events[si].columns.isin(L_cell_names)]
                ca_mats.append(layer_ca_events.to_numpy(dtype=float).T) # filter for cell cols
            
        else:
            ca_mats = []
            for si in range(len(ca_sess_idxs)):
                ca_events[si].replace(to_replace=' nan',value=np.nan,inplace=True) # replace nulls with nans
                ca_mats.append((ca_events[si].filter(regex='C').to_numpy(dtype=float)).T) # filter for cell cols

        ca_mat_all = np.hstack(ca_mats) # no. cells x time steps
        zca = zscore(ca_mat_all,axis=1,nan_policy='omit') # z score across trials for each cell (across cols for each row)
        zca= np.where(np.isnan(zca),0,zca)

        # get time points of cues 
        ca_cue_pulses_ = cms.pulse_to_ca_pulse(cue_pulses,ca_events,ca_sess_idxs)
        ca_cue_pulses = {}
        for cue in ca_cue_pulses_:
            ca_cue_pulses[cue]=[]
        sess_tot = 0
        for sii,si in enumerate(ca_sess_idxs):
            for cue in ca_cue_pulses:
                if np.size(ca_cue_pulses_[cue][sii])>0:
                    ca_cue_pulses[cue].append(ca_cue_pulses_[cue][sii]+sess_tot)
            #tracking[si]['time']+=sess_tot/sample_rate
            sess_tot+=np.size(ca_mats[sii],axis=1)
        for cue in ca_cue_pulses:
            ca_cue_pulses[cue] = np.vstack(ca_cue_pulses[cue]) # combine across sessions


        all_pulses = []
        for cue in ca_cue_pulses:
            for interval in ca_cue_pulses[cue]:
                if len(interval)>1:
                    all_pulses.append(interval)

        all_pulses = np.vstack(all_pulses)
        all_pulses = all_pulses[all_pulses[:,0].argsort()]

        sep = int(np.mean(all_pulses[:,1] - all_pulses[:,0]))

        off_times = []

        for ii in range(len(all_pulses)-1):
            if (all_pulses[ii+1][0] - all_pulses[ii][1])/sep < 5:
                continue
            off_times.append([all_pulses[ii+1][0] - 3*sep, all_pulses[ii+1][0] - 2*sep])
            if (all_pulses[ii+1][0] - all_pulses[ii][1])/sep > 9:
                off_times.append([all_pulses[ii+1][0] - 7*sep, all_pulses[ii+1][0] - 6*sep])

        ca_cue_pulses['off'] = np.array(off_times)

        if correct_trials:
            # Behavioural information
            tracking = cms.load_tracking(b)
            dispenser = np.load(b+'.dispenser',allow_pickle=True)

            # correct trial indices
            speed_scores = [[],[]]
            for cuei,cue in enumerate(cue_pulse_names):
                for si in range(len(stages)):
                    if si in ca_sess_idxs:
                        speed_score = np.zeros(np.size(cue_pulses[cue][si],axis=0))
                        for triali in range(np.size(cue_pulses[cue][si],axis=0)):
                            smpsa =  np.flatnonzero(tracking[si]['time']*sample_rate>cue_pulses[cue][si][triali,1])
                            smpsb = np.flatnonzero(tracking[si]['time']*sample_rate<cue_pulses[cue][si][triali,1]+10*sample_rate)
                            smps = np.intersect1d(smpsa,smpsb)
                            speed_score[triali] = np.sum((tracking[si]['speed'][smps]<1)&(dispenser['near_disp'][si][smps]))
                        speed_scores[cuei].append(speed_score)
            s_scores= np.hstack(speed_scores[0])
            w_scores = np.hstack(speed_scores[1])
            correct = {cue_pulse_names[0]: s_scores>0, cue_pulse_names[1]: w_scores==0}
            incorrect = {cue_pulse_names[0]: s_scores==0, cue_pulse_names[1]: w_scores>0}

        # collect ca activities for each cue
        cue_pop = {}
        for cue in ca_cue_pulses:
            no_time = np.zeros((len(ca_cue_pulses[cue]),n_ca_cells))
            for triali, trial in enumerate(ca_cue_pulses[cue]):
                if shift:
                    no_time[triali] = np.mean(zca[:,trial[1]:trial[1]+(trial[1] - trial[0])],axis=1)
                else:
                    no_time[triali] = np.mean(zca[:,trial[0]:trial[1]],axis=1)
            if (correct_trials == 'correct')&(cue!= 'off'):
                cue_pop[cue] = no_time[correct[cue],: ] # avg across time period, taking only correct trials
            elif (correct_trials == 'incorrect')&(cue!= 'off'):
                cue_pop[cue] = no_time[incorrect[cue],: ]
            else:
                cue_pop[cue] = no_time

        classiefier_pop = {}      
        classiefier_pop['on'] = np.vstack([cue_pop[cue_pulse_names[0]], cue_pop[cue_pulse_names[1]]])
        classiefier_pop['off'] = cue_pop['off']

        return classiefier_pop

def on_off_cells(mouse_ids, rec_days, cue_pulse_names, correct_choice, layer, sess_type = 'tt'):
    
    """ Classifies cells that are characteristic of tone responses relative to off periods.
    Args
        mouse_ids: list of mouse ids as strings in form mhb < mouse id number >
        rec_days: dictionary of mouse ids and list of recording days
        correct_choice: whether to filter for correct/incorrect trials. Either correct, incorrect or False 
        sess_type: type of trial. Either 'tt','al' or 'rl'
    Returns
        a dataframe with a list of boolean values indicating which cells are active to tones for each mouse and recording day """
    
    res = {}
    for mouse_id in mouse_ids:
        
        day_res = {}
        for rr, rec_day in enumerate(rec_days[mouse_id]):
            
            layer_res = {}
            print(mouse_id , '  ', rec_day)
            cue_pop = on_off_cuepop(mouse_id, rec_day, cue_pulse_names, correct_choice, layer, sess_type)
            
            X, Y, Y_string = extract_XY(cue_pop)
            unique, counts = np.unique(Y, return_counts = True)

            if (len(unique)<2) or (min(counts)<2) or (X.shape[1] < 1):
                do_clf=0
                print('skipped : ', mouse_id, '  ', rec_day)
            else:
                do_clf=1

            if do_clf:
                classes = list(cue_pop.keys())
                coefs = []
                cv = RepeatedStratifiedKFold(n_splits=np.min(counts),n_repeats=10)

                for i, (train_index, test_index) in enumerate(cv.split(X,Y)):
                    X_test, X_train, Y_test, Y_train  = X[test_index], X[train_index], Y[test_index], Y[train_index]

                    # scale and transform
                    scaling = StandardScaler()
                    scaling.fit(X_train)

                    # Transform training and test
                    X_test_scaled = scaling.transform(X_test)

                    # fit model
                    model =  LogisticRegression(penalty='l1',solver='liblinear', random_state=0).fit(X_train,Y_train)
                    score = balanced_accuracy_score(Y_test, model.predict(X_test_scaled), adjusted=True)
                    #extract coefficients
                    coefs.append(model.coef_)

                # identify tone responsive cells
                on_cells = np.mean(coefs, axis=0) > 0 # boolean vector of active cell indices

                day_res[rec_day] = on_cells
            res[mouse_id] = day_res 
    return res

  