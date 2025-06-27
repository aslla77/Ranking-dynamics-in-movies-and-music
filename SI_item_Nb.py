import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pycountry
import pickle

max_PDF=1 # Highest rank PDF
first_PDF=1 # first rank PDF
last_PDF=1 # last rank PDF
last_first_PDF=1 # first and last rank PDF
co=1 # PCC(pearsonn correlation)
# ========================================================================================

load_data='./data/'
save_result='./result/SI_Nb/'
save_data='./figure/SI_Nb/'
A=['ec', 'fr', 'ar', 'fi', 'no', 'it', 'ph', 'tw', 'nz', 'tr', 'us', 'sv', 'cr', 'de', 'cl', 'jp', 'br', 'hn', 'gt', 'ch', 'hu', 'ca', 'pe', 'be', 'dk', 'bo', 'pl', 'at', 'pt', 'se', 'mx', 'pa', 'uy', 'is', 'es', 'cz', 'ie', 'nl', 'co', 'sg', 'id', 'do', 'gb', 'py', 'au', 'gr', 'hk']
ott_start='2023-01-01'
spotify_start='2017-01-01'


def label_setting(A,B,s=40):
    plt.axes(A)
    font = {'weight': 'bold',
        'size': s}
    plt.text(0., 0., B, fontdict=font)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)


def save_fig(A):
    plt.savefig(save_data+A+'.png',bbox_inches='tight')
    plt.savefig(save_result+A+'.pdf',bbox_inches='tight')
    # ,transparent=True    


# # # ======================================================================================================================================================================
# # # =============item part  (cut data with N_b)!!!!!========================================================================================================================
# # # ======================================================================================================================================================================

import boxp
with open(file='./ott2/data/netflix_Nb.pickle', mode='rb') as f:
    Netflix=pickle.load(f)

with open(file='./ott2/data/hbo_Nb.pickle', mode='rb') as f:
    Hbo=pickle.load(f)

with open(file='./ott2/data/disney_Nb.pickle', mode='rb') as f:
    Disney=pickle.load(f)

with open(file='./ott2/data/prime_Nb.pickle', mode='rb') as f:
    Prime=pickle.load(f)

with open(file='./ott2/data/tw_Nb.pickle', mode='rb') as f:
    Tw=pickle.load(f)

with open(file='./ott2/data/gb_Nb.pickle', mode='rb') as f:
    UK=pickle.load(f)

with open(file='./ott2/data/us_Nb.pickle', mode='rb') as f:
    USA=pickle.load(f)

with open(file='./ott2/data/au_Nb.pickle', mode='rb') as f:
    AU=pickle.load(f)
data_list1=[Netflix,Disney,Hbo,Prime,Tw,UK,USA,AU]
data_name=["netflix",'Disney','Hbo','prime','tw_spotify','gb_spotify','us_spotify','au_spotify']


def fig_7_set(A,C,x_label=False,y_label=False):
  plt.axes(A)
  boxp.Max_binning(C,20,True,False,x_label,y_label)
  
  ax=plt.gca()
  for spine in ax.spines.values():
      spine.set_linewidth(3)
  # plt.yticks([])
    
  plt.tick_params(width=3,  length=5, pad=6, labelsize=30)
  
if max_PDF:
# #@title it_barplot
    title_font = {    'fontsize': 30}

    x=np.arange(0.01,10,0.1)
    plt.figure(figsize=(24,24))
    #aplot (Netflix)
    A=[0.05, 0.775, 0.425, 0.205]
    fig_7_set(A,Netflix,False,1)

    #bplot (Disney)
    A=[0.55, 0.775, 0.425, 0.205]
    fig_7_set(A,Hbo)

    #cplot (Disney+)
    A=[0.05, 0.525, 0.425, 0.205]
    fig_7_set(A,Disney,False,1)

    #dplot (prime)
    A=[0.55, 0.525, 0.425, 0.205]
    fig_7_set(A,Prime)

    #E
    A=[0.05, 0.275, 0.425, 0.205]
    fig_7_set(A,Tw,False,1)

    #F
    A=[0.55, 0.275, 0.425, 0.205]
    fig_7_set(A,UK)

    #G
    A=[0.05, 0.025, 0.425, 0.205]
    fig_7_set(A,USA,1,1)

    #H
    A=[0.55, 0.025, 0.425, 0.205]
    fig_7_set(A,AU,1)


    s=30
    label_setting([0.01, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.989, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.739, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.489, 0.01, 0.01],'E. Spotify Taiwan',s)   # A=[0.05, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.489, 0.01, 0.01],'F. Spotify UK',s)   # A=[0.28, 0.1, 0.2, 0.32])
    label_setting([0.01, 0.239, 0.01, 0.01],'G. Spotify USA',s)   # A=[0.51, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.239, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

    save_fig('SI_PDF_max_rank_Nb')
    plt.close()
    print('SI_PDF_max_rank_Nb')

def fig_7_set(A,C,x_label=False,y_label=False):
  plt.axes(A)
  boxp.first_binning(C,20,True,False,x_label,y_label)
  
  ax=plt.gca()
  for spine in ax.spines.values():
      spine.set_linewidth(3)
  # plt.yticks([])
    
  plt.tick_params(width=3,  length=5, pad=6, labelsize=30)

if first_PDF:
    title_font = {    'fontsize': 30}

    x=np.arange(0.01,10,0.1)
    plt.figure(figsize=(24,24))
    #aplot (Netflix)
    A=[0.05, 0.775, 0.425, 0.205]
    fig_7_set(A,Netflix,False,1)

    #bplot (Disney)
    A=[0.55, 0.775, 0.425, 0.205]
    fig_7_set(A,Hbo)

    #cplot (Disney+)
    A=[0.05, 0.525, 0.425, 0.205]
    fig_7_set(A,Disney,False,1)

    #dplot (prime)
    A=[0.55, 0.525, 0.425, 0.205]
    fig_7_set(A,Prime)

    #E
    A=[0.05, 0.275, 0.425, 0.205]
    fig_7_set(A,Tw,False,1)

    #F
    A=[0.55, 0.275, 0.425, 0.205]
    fig_7_set(A,UK)

    #G
    A=[0.05, 0.025, 0.425, 0.205]
    fig_7_set(A,USA,1,1)

    #H
    A=[0.55, 0.025, 0.425, 0.205]
    fig_7_set(A,AU,1)


    s=30
    label_setting([0.01, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.989, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.739, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.489, 0.01, 0.01],'E. Spotify Taiwan',s)   # A=[0.05, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.489, 0.01, 0.01],'F. Spotify UK',s)   # A=[0.28, 0.1, 0.2, 0.32])
    label_setting([0.01, 0.239, 0.01, 0.01],'G. Spotify USA',s)   # A=[0.51, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.239, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

    save_fig('SI_PDF_first_rank_Nb')
    plt.close()
    print('SI_PDF_first_rank_Nb')

def fig_7_set(A,C,x_label=False,y_label=False):
  plt.axes(A)
  boxp.last_binning(C,20,True,False,x_label,y_label)
  
  ax=plt.gca()
  for spine in ax.spines.values():
      spine.set_linewidth(3)
    
  plt.tick_params(width=3,  length=5, pad=6, labelsize=30)

if last_PDF:
    title_font = {    'fontsize': 30}

    x=np.arange(0.01,10,0.1)
    plt.figure(figsize=(24,24))
    #aplot (Netflix)
    A=[0.05, 0.775, 0.425, 0.205]
    fig_7_set(A,Netflix,False,1)

    #bplot (Disney)
    A=[0.55, 0.775, 0.425, 0.205]
    fig_7_set(A,Hbo)

    #cplot (Disney+)
    A=[0.05, 0.525, 0.425, 0.205]
    fig_7_set(A,Disney,False,1)

    #dplot (prime)
    A=[0.55, 0.525, 0.425, 0.205]
    fig_7_set(A,Prime)

    #E
    A=[0.05, 0.275, 0.425, 0.205]
    fig_7_set(A,Tw,False,1)

    #F
    A=[0.55, 0.275, 0.425, 0.205]
    fig_7_set(A,UK)

    #G
    A=[0.05, 0.025, 0.425, 0.205]
    fig_7_set(A,USA,1,1)

    #H
    A=[0.55, 0.025, 0.425, 0.205]
    fig_7_set(A,AU,1)


    s=30
    label_setting([0.01, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.989, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.739, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.489, 0.01, 0.01],'E. Spotify Taiwan',s)   # A=[0.05, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.489, 0.01, 0.01],'F. Spotify UK',s)   # A=[0.28, 0.1, 0.2, 0.32])
    label_setting([0.01, 0.239, 0.01, 0.01],'G. Spotify USA',s)   # A=[0.51, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.239, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

    save_fig('SI_PDF_last_rank_Nb')
    plt.close()
    print('SI_PDF_last_rank_Nb')

def fig_8_set(A,C,x_label=False,y_label=False):
    plt.axes(A)
    boxp.first_binning(C,20,True,False,x_label,y_label,'blue')
    boxp.last_binning1(C,20,True,False,x_label,y_label)
    if x_label:
        plt.xlabel("Rank",fontsize=45)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # plt.yticks([])
        
    plt.tick_params(width=3,  length=5, pad=6, labelsize=30)

if last_first_PDF:
    title_font = {    'fontsize': 30}

    x=np.arange(0.01,10,0.1)
    plt.figure(figsize=(24,24))
    #aplot (Netflix)
    A=[0.05, 0.775, 0.425, 0.205]
    fig_8_set(A,Netflix,False,1)

    #bplot (Disney)
    A=[0.55, 0.775, 0.425, 0.205]
    fig_7_set(A,Hbo)
    fig_8_set(A,Hbo)
    line1, = plt.plot([],[],linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor='red', # 마커 내부 색깔
    markeredgecolor='red',
    lw=3,
    ms=10)

    line2, = plt.plot([],[],linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor='blue', # 마커 내부 색깔
    markeredgecolor='blue',
    lw=3,
    ms=10)

    plt.legend([ line2,line1], [ 'First Rank','Last rank'],fontsize=35)

    #cplot (Disney+)
    A=[0.05, 0.525, 0.425, 0.205]
    fig_8_set(A,Disney,False,1)

    #dplot (prime)
    A=[0.55, 0.525, 0.425, 0.205]
    fig_8_set(A,Prime)

    #E
    A=[0.05, 0.275, 0.425, 0.205]
    fig_8_set(A,Tw,False,1)

    #F
    A=[0.55, 0.275, 0.425, 0.205]
    fig_8_set(A,UK)

    #G
    A=[0.05, 0.025, 0.425, 0.205]
    fig_8_set(A,USA,1,1)

    #H
    A=[0.55, 0.025, 0.425, 0.205]
    fig_8_set(A,AU,1)


    s=30
    label_setting([0.01, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.989, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
    label_setting([0.51, 0.739, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
    label_setting([0.01, 0.489, 0.01, 0.01],'E. Spotify Taiwan',s)   # A=[0.05, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.489, 0.01, 0.01],'F. Spotify UK',s)   # A=[0.28, 0.1, 0.2, 0.32])
    label_setting([0.01, 0.239, 0.01, 0.01],'G. Spotify USA',s)   # A=[0.51, 0.1, 0.2, 0.32])
    label_setting([0.51, 0.239, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

    save_fig('SI_PDF_last_first_Nb')
    plt.close()
    print('SI_PDF_last_first_Nb')


from scipy import stats
from datetime import datetime
def corr(C,dname):

    x0=[]
    x1=[]
    x2=[]
    x3=[]

    for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
        if (C[i][0]!=0 and C[i][1]!=np.inf and C[i][2]!=np.inf) and not(C[i][6]==C['E']):
            s=datetime.strptime(C[i][5], '%Y-%m-%d')
            e=datetime.strptime(C[i][6], '%Y-%m-%d')
            day_gap=e-s
            x0.append(day_gap.days)
            x1.append(C[i][1])
            x2.append(C[i][2])
            x3.append(C[i][3])

    print(dname)
    print('life time - first rank :',stats.pearsonr(x0,x1))
    print('life time - higest rank :',stats.pearsonr(x0,x2))
    print('life time - last rank :',stats.pearsonr(x0,x3))
    print('first rank - higest rank :',stats.pearsonr(x1,x2))
    print('first rank - last rank :',stats.pearsonr(x1,x3))
    print('last rank - higest rank :',stats.pearsonr(x2,x3))
    
if co:
    for i in range(len(data_list1)):
        corr(data_list1[i],data_name[i])