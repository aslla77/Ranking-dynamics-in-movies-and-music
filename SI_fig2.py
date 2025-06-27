
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pycountry
import pickle
import os


N_b_c=0
alpha_c=0
N_b_fre=0
alpha_fre=0
co=0
rate=0


xlabelpad=20
xtick_size=35

load_data='./data/'
save_result='./result/SI1/'
save_data='./figure/SI1/'

path = '/your/path/here'

if os.path.exists(save_result):
    print("exists path.")
else:
    print("no exists path.")
    os.makedirs(save_result, exist_ok=True)

if os.path.exists(save_data):
    print("exists path.")
else:
    print("no exists path.")
    os.makedirs(save_data, exist_ok=True)



A=['ec', 'fr', 'ar', 'fi', 'no', 'it', 'ph', 'tw', 'nz', 'tr', 'us', 'sv', 'cr', 'de', 'cl', 'jp', 'br', 'hn', 'gt', 'ch', 'hu', 'ca', 'pe', 'be', 'dk', 'bo', 'pl', 'at', 'pt', 'se', 'mx', 'pa', 'uy', 'is', 'es', 'cz', 'ie', 'nl', 'co', 'sg', 'id', 'do', 'gb', 'py', 'au', 'gr', 'hk']
ott_start='2023-01-01'
spotify_start='2017-01-01'
# from google.colab import drive
# drive.mount('/content/drive')

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


class setting():
   def __init__(self,name):
    self.fluc=np.load(load_data+name+'_flux.npy')
    self.In_SCI=np.load(load_data+name+'_Interflux.npy')
    self.out_SCI=np.load(load_data+name+'_outflux.npy')
    self.enter_SCI=np.load(load_data+name+'_Influx.npy')
    self.frequncy=np.load(load_data+name+'_f.npy')
    self.pod=np.load(load_data+name+'_p.npy')
    self.new_o_list=np.load(load_data+name+'_newo.npy')

    self.pdf_out=np.load(load_data+name+'_pdf_out.npy')
    self.pdf_in=np.load(load_data+name+'_pdf_in.npy')
    self.pdf_bin=np.load(load_data+name+'_pdf_bin.npy')

    self.cdf_out=np.load(load_data+name+'_cdf_out.npy')
    self.cdf_in=np.load(load_data+name+'_cdf_in.npy')
    self.cdf_bin=np.load(load_data+name+'_cdf_bin.npy')
    self.F_list=np.load(load_data+name+'_F_list.npy')
    self.down=np.load(load_data+name+'_down.npy')
    self.up=np.load(load_data+name+'_up.npy')
    self.N_0=np.load(load_data+name+'_N.npy')
    self.T=len(self.fluc)
    print(self.T)

    self.matrix=np.load(load_data+name+'_matrix.npy')
    self.Nlast=np.load(load_data+name+'_Nlast.npy')
    self.rb=np.load(load_data+name+'_rb.npy')
    self.rf=np.load(load_data+name+'_rf.npy')
    self.fx=np.load(load_data+name+'_fx.npy')
    self.prob=np.load(load_data+name+'_prob.npy')
    self.prob_1=np.load(load_data+name+'_prob1.npy')

    self.beta_0_in=np.load(load_data+name+'_beta_0_in.npy')
    self.beta_0_out=np.load(load_data+name+'_beta_0_out.npy')
    self.beta_0_inter=np.load(load_data+name+'_beta_0_inter.npy')
    self.beta_1_in=np.load(load_data+name+'_beta_1_in.npy')
    self.beta_1_out=np.load(load_data+name+'_beta_1_out.npy')
    self.beta_1_inter=np.load(load_data+name+'_beta_1_inter.npy')
    self.beta_m1_in=np.load(load_data+name+'_beta_m1_in.npy')
    self.beta_m1_out=np.load(load_data+name+'_beta_m1_out.npy')
    self.beta_m1_inter=np.load(load_data+name+'_beta_m1_inter.npy')
    # self.measure_data=np.load(load_data+name+'_measure_data.npy')

    if not(name in ['netflix','Disney','Hbo','prime']):
      self.dsci=np.load(load_data+name+'_dSCI.npy')
      self.dr=np.load(load_data+name+'_dr.npy')
      self.ds=np.load(load_data+name+'_ds.npy')
      self.s_list=np.load(load_data+name+'_ds_list.npy')
      self.r_list=np.load(load_data+name+'_dr_list.npy')
      self.r_array=np.load(load_data+name+'_r_array.npy')
      self.s_array=np.load(load_data+name+'_s_array.npy')
      self.intra_dr=np.load(load_data+name+'_dr_intra.npy')
      self.out_dr=np.load(load_data+name+'_dr_out.npy')
      self.enter_dr=np.load(load_data+name+'_dr_enter.npy')
    else:
      self.dr=np.load(load_data+name+'_dr.npy')
      self.intra_dr=np.load(load_data+name+'_dr_intra.npy')
      self.out_dr=np.load(load_data+name+'_dr_out.npy')
      self.enter_dr=np.load(load_data+name+'_dr_enter.npy')



#데이터 불러오기
#
netflix=setting('netflix')
Disney=setting('Disney')
Hbo=setting('Hbo')
prime=setting('prime')
tw_spotify=setting('tw_spotify')
us_spotify=setting('us_spotify')
au_spotify=setting('au_spotify')
gb_spotify=setting('gb_spotify')
# sw_spotify=setting('sw_spotify')
# jp_spotify=setting('jp_spotify')



def save_fig(A):
    plt.savefig(save_data+A+'.png',bbox_inches='tight')
    plt.savefig(save_result+A+'.pdf',bbox_inches='tight')
    # ,transparent=True    





def N_b_change(A,B,name,N_b,s_date,e_date,X,Y,xtick=None,ytick=None):
    plt.axes(A)
    per=np.array([0,0.25,0.5,0.75,1])
    c=False
    alpha=0
    alpha=B.N_0+int(B.N_0*alpha)+1
    # print(N_b,alpha)
#   tot_SCI,abs_SCI,enter_SCI,out_SCI,in    side_SCI=A.max_min(N_b,alpha)
    tot_SCI=np.load(load_data+name+f'tot_{N_b}_{alpha}.npy')
    abs_SCI=np.load(load_data+name+f'abs_{N_b}_{alpha}.npy')
    enter_SCI=np.load(load_data+name+f'enter_{N_b}_{alpha}.npy')
    out_SCI=np.load(load_data+name+f'out_{N_b}_{alpha}.npy')
    inside_SCI=np.load(load_data+name+f'intra_{N_b}_{alpha}.npy')
    enter_SCI=abs(enter_SCI)
    out_SCI=abs(out_SCI)
    inside_SCI=abs(inside_SCI)

    if s_date==None and e_date==None:
        arang=B.T
        p1 = plt.bar(np.arange(arang), enter_SCI, color='green')

        p2 = plt.bar(np.arange(arang), inside_SCI, color='dodgerblue',
        bottom=enter_SCI) # stacked bar chart

        p3 = plt.bar(np.arange(arang),out_SCI, color='orange',
        bottom=enter_SCI+inside_SCI)
        

    else:
        c=True
        arang=e_date-s_date
        p1 = plt.bar(np.arange(arang), enter_SCI[s_date:e_date], color='green')

        p2 = plt.bar(np.arange(arang), inside_SCI[s_date:e_date], color='dodgerblue',
                    bottom=enter_SCI[s_date:e_date]) # stacked bar chart

        p3 = plt.bar(np.arange(arang),out_SCI[s_date:e_date], color='orange',
                    bottom=enter_SCI[s_date:e_date]+inside_SCI[s_date:e_date])

    
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
  # plt.yticks([])
    if X!= None:
        plt.xlabel('$t$ (day)',labelpad=xlabelpad,fontdict={         'size': 50})
    if Y!= None:
        plt.ylabel('$c_t$',fontdict={         'size': 50},loc='center',labelpad=15)
    if ytick!=None:
        plt.yticks([])
    else:
        ax=plt.gca()
        ax.set_xticks(np.arange(0,max(abs_SCI),50))
        plt.tick_params(width=3,  length=5, pad=6, labelsize=35)
    if xtick!=None:
        plt.xticks([])
    else:
        ax=plt.gca()
        ax.set_xticks(np.arange(0,arang,30))
        plt.tick_params(width=3,  length=5, pad=6, labelsize=35)
  
    if c and arang>=int(B.T/2):
        plt.xlim(0,arang)
    else:
        plt.xlim(0,arang)
    

       

def alpha_change(A,B,name,alpha,s_date,e_date,X,Y,xtick=None,ytick=None):
    plt.axes(A)
    per=np.array([0,0.25,0.5,0.75,1])
    c=False
    N_b=int(B.N_0)
    # print(N_b,alpha)
#   tot_SCI,abs_SCI,enter_SCI,out_SCI,inside_SCI=A.max_min(N_b,alpha)
    tot_SCI=np.load(load_data+name+f'tot_{N_b}_{alpha}.npy')
    abs_SCI=np.load(load_data+name+f'abs_{N_b}_{alpha}.npy')
    enter_SCI=np.load(load_data+name+f'enter_{N_b}_{alpha}.npy')
    out_SCI=np.load(load_data+name+f'out_{N_b}_{alpha}.npy')
    inside_SCI=np.load(load_data+name+f'intra_{N_b}_{alpha}.npy')
    enter_SCI=abs(enter_SCI)
    out_SCI=abs(out_SCI)
    inside_SCI=abs(inside_SCI)
    
    if s_date==None and e_date==None:
        arang=B.T
        p1 = plt.bar(np.arange(arang), enter_SCI, color='green')

        p2 = plt.bar(np.arange(arang), inside_SCI, color='dodgerblue',
        bottom=enter_SCI) # stacked bar chart

        p3 = plt.bar(np.arange(arang),out_SCI, color='orange',
        bottom=enter_SCI+inside_SCI)
        

    else:
        c=True
        arang=e_date-s_date
        p1 = plt.bar(np.arange(arang), enter_SCI[s_date:e_date], color='green')

        p2 = plt.bar(np.arange(arang), inside_SCI[s_date:e_date], color='dodgerblue',
                    bottom=enter_SCI[s_date:e_date]) # stacked bar chart

        p3 = plt.bar(np.arange(arang),out_SCI[s_date:e_date], color='orange',
                    bottom=enter_SCI[s_date:e_date]+inside_SCI[s_date:e_date])

    
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
  # plt.yticks([])
    if X!= None:
        plt.xlabel('$t$ (day)',labelpad=xlabelpad,fontdict={         'size': 50})
    if Y!= None:
        plt.ylabel('$c_t$',fontdict={         'size': 50},loc='center',labelpad=15)
    if ytick!=None:
        plt.yticks([])
    else:
        ax=plt.gca()
        ax.set_xticks(np.arange(0,max(abs_SCI),50))
        plt.tick_params(width=3,  length=5, pad=6, labelsize=35)
    if xtick!=None:
        plt.xticks([])
        
    else:
        ax=plt.gca()
        ax.set_xticks(np.arange(0,arang,30))
        plt.tick_params(width=3,  length=5, pad=6, labelsize=35)
  
    if c and arang>=int(B.T/2):
        plt.xlim(0,arang)
    else:
        plt.xlim(0,arang)

def per_check(B,N_b,alpha,name):
    tot_SCI=np.load(load_data+name+f'tot_{N_b}_{alpha}.npy')
    abs_SCI=np.load(load_data+name+f'abs_{N_b}_{alpha}.npy')
    enter_SCI=np.load(load_data+name+f'enter_{N_b}_{alpha}.npy')
    out_SCI=np.load(load_data+name+f'out_{N_b}_{alpha}.npy')
    inside_SCI=np.load(load_data+name+f'intra_{N_b}_{alpha}.npy')
    enter_SCI=abs(enter_SCI)
    out_SCI=abs(out_SCI)
    inside_SCI=abs(inside_SCI)
    in_per=np.sum(enter_SCI)/np.sum(abs_SCI)
    out_per=np.sum(out_SCI)/np.sum(abs_SCI)
    intra_per=np.sum(inside_SCI)/np.sum(abs_SCI)

    return in_per,out_per,intra_per

def fre_n_alpha(A,B,name,N_b,alpha,s_date,e_date,X,Y,xtick=None,ytick=None):
    plt.axes(A)
    per=np.array([0,0.25,0.5,0.75,1])
    print(N_b,alpha)
#   tot_SCI,abs_SCI,enter_SCI,out_SCI,inside_SCI=A.max_min(N_b,alpha)
    tot_SCI=np.load(load_data+name+f'tot_{N_b}_{alpha}.npy')
    abs_SCI=np.load(load_data+name+f'abs_{N_b}_{alpha}.npy')
    enter_SCI=np.load(load_data+name+f'enter_{N_b}_{alpha}.npy')
    out_SCI=np.load(load_data+name+f'out_{N_b}_{alpha}.npy')
    inside_SCI=np.load(load_data+name+f'intra_{N_b}_{alpha}.npy')
    enter_SCI=abs(enter_SCI)
    out_SCI=abs(out_SCI)
    inside_SCI=abs(inside_SCI)

    plt.axvline(365/7, linestyle='--', linewidth=2 , c='red')

    f, P = scipy.signal.periodogram(abs_SCI, B.T, nfft=2**12)
    # plt.plot(f, P,c='#536512',linewidth=2,label=label1)
    plt.plot(f, P,c='purple',linewidth=2)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
  # plt.yticks([])
    if X!= None:
        plt.xlabel('Frequency (1/day)',labelpad=xlabelpad,fontdict={         'size': 50})
        fig_4_tick(xtick)
    if Y!= None:
        plt.ylabel('Power Spectrum',fontdict={         'size': 30},loc='center',labelpad=10)
    if ytick!=None:
        plt.yticks([])
    else:
        ax=plt.gca()
        fig_4_tick(xtick)
        plt.tick_params(width=3,  length=5, pad=6, labelsize=25)


X_label='Frequency (1/day)'
# X_TICK_LABEL=['365', '14','7','5','4','3','2']
# X_TICK_LABEL=['365', '7','4','3','2']
X_TICK_LABEL=['1/365', '1/7','1/4','1/3','1/2']
def fig_4_tick(X):
    #그래프 정리
    plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        ax.patch.set_alpha(0)

    plt.yticks([])

    # ax.set_xticks([0,365/14,365/7,365/5,365/4,365/3,365/2])
    ax.set_xticks([0,365/7,365/4,365/3,365/2])
    ax.set_xticklabels(X_TICK_LABEL, fontsize=20)
    plt.tick_params(width=3,  length=5, pad=6, labelsize=20)  



# # # # ======================================================================================================================================================================
data_list=[netflix,Disney,Hbo,prime,tw_spotify,gb_spotify,us_spotify,au_spotify]
data_name=["netflix",'Disney','Hbo','prime','tw_spotify','gb_spotify','us_spotify','au_spotify']

# # # ======================================================================================================================================================================
def fre_n_alpha(A,B,name,N_b,alpha,X,Y,xtick=None,ytick=None,count=0):
    per=np.array([0,0.25,0.5,0.75,1])
    print(N_b,alpha)
#   tot_SCI,abs_SCI,enter_SCI,out_SCI,inside_SCI=A.max_min(N_b,alpha)
    tot_SCI=np.load(load_data+name+f'tot_{N_b}_{alpha}.npy')
    abs_SCI=np.load(load_data+name+f'abs_{N_b}_{alpha}.npy')
    enter_SCI=np.load(load_data+name+f'enter_{N_b}_{alpha}.npy')
    out_SCI=np.load(load_data+name+f'out_{N_b}_{alpha}.npy')
    inside_SCI=np.load(load_data+name+f'intra_{N_b}_{alpha}.npy')
    enter_SCI=abs(enter_SCI)
    out_SCI=abs(out_SCI)
    inside_SCI=abs(inside_SCI)

    

    f, P = scipy.signal.periodogram(abs_SCI, B.T, nfft=2**12)
    # plt.plot(f, P,c='#536512',linewidth=2,label=label1)
    if count==0:
        plt.plot(f, P,linewidth=2,label=fr'$N_0$={N_b}',alpha=0.3)
    elif count!=0:
        plt.plot(f, P,linewidth=2,label=fr'$\alpha$={alpha-B.N_0}',alpha=0.3)

    plt.yticks([])
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('')


    

# A,B,name,N_b,alpha,s_date,e_date,X,Y,xtick=None,ytick=None
def SI_fig_fre_N_B(A,B,data_name,X,Y,xtick=None,ytick=None):
    plt.axes(A)
    r=[1,0.75,0.5,0.25]
    print(data_name)
    plt.axvline(365/7, linestyle='--', linewidth=2 , c='red',alpha=0.3)
    count=0
    for i in r:   
        N_b=int(B.N_0*i)
        alpha=N_b+1
        fre_n_alpha(A,B,data_name,N_b,alpha,X,Y,xtick,ytick,count)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
  # plt.yticks([])
    if X!= None:
        plt.xlabel('Frequency (1/day)',labelpad=xlabelpad,fontdict={         'size': 50})
    if xtick!=None:    
        fig_4_tick(xtick)
    if Y!= None:
        plt.ylabel('Power Spectrum',fontdict={         'size': 50},loc='center',labelpad=10)
    if ytick!=None:
        plt.yticks([])
    else:
        ax=plt.gca()
        fig_4_tick(xtick)
    plt.tick_params(width=3,  length=5, pad=6, labelsize=35)
    legend = plt.legend(fontsize=20)
    legend.get_frame().set_alpha(0.3) 
    

def SI_fig_fre_alpha(A,B,data_name,Xlabel,Ylabel,xtick=None,ytick=None):
    plt.axes(A)
    # r=[0,0.25,0.5,0.75,1]
    r=[1,0.75,0.5,0.25,0]
    print(data_name)
    N_b=B.N_0
    plt.axvline(365/7, linestyle='--', linewidth=2 , c='red',alpha=0.3)
    count=1
    for i in r:   
        alpha=B.N_0+1+int(B.N_0*i)
        fre_n_alpha(A,B,data_name,N_b,alpha,Xlabel,Ylabel,xtick,ytick,count)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # plt.yticks([])
    print('label:',Xlabel)
    if Xlabel:
        plt.xlabel('Frequency (1/day)',labelpad=xlabelpad,fontdict={         'size': 50})
    if xtick!=None:    
        fig_4_tick(xtick)
    if Ylabel!= None:
        plt.ylabel('Power Spectrum',fontdict={         'size': 50},loc='center',labelpad=10)
    if ytick!=None:
        plt.yticks([])
    else:
        ax=plt.gca()
        fig_4_tick(xtick)
    plt.tick_params(width=3,  length=5, pad=6, labelsize=35)
    legend = plt.legend(fontsize=20)
    legend.get_frame().set_alpha(0.3) 

A_list=[]
s=30
if N_b_fre:
    plt.figure(figsize=(35,19))
    A=[0.05, 0.59, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[0],data_name[0],None,1,1,1)
    label_setting([0.03, 0.919, 0.01, 0.01],'A.Netflix',s)  

    A=[0.28, 0.59, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[1],data_name[1],None,None,1,1)
    label_setting([0.26, 0.919, 0.01, 0.01],'B.HBO',s) 

    A=[0.51, 0.59, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[2],data_name[2],None,None,1,1)
    label_setting([0.49, 0.919, 0.01, 0.01],'C.Disney+',s) 

    A=[0.74, 0.59, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[3],data_name[3],None,None,1,1)
    label_setting([0.72, 0.919, 0.01, 0.01],'D.Amazon Prime',s) 

    A=[0.05, 0.19, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[4],data_name[4],1,1,1,1)
    label_setting([0.03, 0.519, 0.01, 0.01],'E.Spotify Taiwan',s) 

    A=[0.28, 0.19, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[5],data_name[5],1,None,1,1)
    label_setting([0.26, 0.519, 0.01, 0.01],'F.Spotify UK',s)

    A=[0.51, 0.19, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[6],data_name[6],1,None,1,1)
    label_setting([0.49, 0.519, 0.01, 0.01],'G.Spotify USA',s)

    A=[0.74, 0.19, 0.2, 0.32]
    SI_fig_fre_N_B(A,data_list[7],data_name[7],1,None,1,1)
    label_setting([0.72, 0.519, 0.01, 0.01],'H.Spotify Australia',s)

    save_fig('SI_N_b_fre')
    plt.close()
    print('SI_N_b_fre')



if alpha_fre:
    # SI_fig_fre_alpha(A,B,data_name,N_b,alpha,X,Y,xtick=None,ytick=None)
    plt.figure(figsize=(35,19))
    A=[0.05, 0.59, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[0],data_name[0],False,1,1,1)
    label_setting([0.03, 0.919, 0.01, 0.01],'A.Netflix',s)  

    A=[0.28, 0.59, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[1],data_name[1],False,None,1,1)
    label_setting([0.26, 0.919, 0.01, 0.01],'B.HBO',s) 

    A=[0.51, 0.59, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[2],data_name[2],False,None,1,1)
    label_setting([0.49, 0.919, 0.01, 0.01],'C.Disney+',s) 

    A=[0.74, 0.59, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[3],data_name[3],False,None,1,1)
    label_setting([0.72, 0.919, 0.01, 0.01],'D.Amazon Prime',s) 

    A=[0.05, 0.19, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[4],data_name[4],1,1,1,1)
    label_setting([0.03, 0.519, 0.01, 0.01],'E.Spotify Taiwan',s) 

    A=[0.28, 0.19, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[5],data_name[5],1,None,1,1)
    label_setting([0.26, 0.519, 0.01, 0.01],'F.Spotify UK',s)

    A=[0.51, 0.19, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[6],data_name[6],1,None,1,1)
    label_setting([0.49, 0.519, 0.01, 0.01],'G.Spotify USA',s)

    A=[0.74, 0.19, 0.2, 0.32]
    SI_fig_fre_alpha(A,data_list[7],data_name[7],1,None,1,1)
    label_setting([0.72, 0.519, 0.01, 0.01],'H.Spotify Australia',s)

    save_fig('SI_alpha_fre')
    plt.close()
    print('SI_alpha_fre')






#====================================================================================
X_label='Frequency (1/day)'
# X_TICK_LABEL=['365', '14','7','5','4','3','2']
# X_TICK_LABEL=['365', '7','4','3','2']
X_TICK_LABEL=['1/365', '1/7','1/4','1/3','1/2']
def biggest_marker(a):
    max_idx = np.argmax(a)     # 최댓값의 인덱스
    max_val = a[max_idx]       # 최댓값
    return max_idx,max_val

def marker_set(locate,B):
    m_size=25
    ax=plt.twinx()
    A=B.fluc
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    max_idx,max_val=biggest_marker(P)
    ax.plot(f[max_idx],10 ,  c='purple',linewidth=3,label='min', marker='D', markerfacecolor='purple', markeredgecolor='black',markersize=m_size)
    ax.set_ylim(0,11)
    ax.yaxis.set_visible(False) 

    ax=plt.twinx()
    A=B.enter_dr+B.out_dr+B.intra_dr
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    max_idx,max_val=biggest_marker(P)
    ax.plot(f[max_idx],8 ,  c='red',linewidth=2,label=r'$\beta=0$', marker='p', markerfacecolor='red', markeredgecolor='black',markersize=m_size)
    ax.set_ylim(0,11)
    ax.yaxis.set_visible(False) 

    ax=plt.twinx()
    A=B.beta_0_in+B.beta_0_out+B.beta_0_inter
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    max_idx,max_val=biggest_marker(P)
    ax.plot(f[max_idx],9 ,  c='brown',linewidth=2,label=r'$\sqrt{r_{t}r_{t-1}}$', marker='P', markerfacecolor='brown', markeredgecolor='black',markersize=m_size)
    ax.set_ylim(0,11)
    ax.yaxis.set_visible(False) 

    ax=plt.twinx()
    A=B.beta_1_in+B.beta_1_out+B.beta_1_inter
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    max_idx,max_val=biggest_marker(P)
    ax.plot(f[max_idx],7 ,  c='pink',linewidth=2,label=r'$\beta=1$', marker='X', markerfacecolor='pink', markeredgecolor='black',markersize=m_size)
    ax.set_ylim(0,11)
    ax.yaxis.set_visible(False) 

    ax=plt.twinx()
    A=B.beta_m1_in+B.beta_m1_out+B.beta_m1_inter
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    max_idx,max_val=biggest_marker(P)
    ax.plot(f[max_idx],6 ,  c='olive',linewidth=2,label=r'$\beta=-1$', marker='o', markerfacecolor='olive', markeredgecolor='black',markersize=m_size)
    ax.set_ylim(0,11)
    ax.yaxis.set_visible(False) 
    



def SI_beta(locate,B,C,X,Y,label1):
    plt.axes(locate)
    A=B.fluc
    plt.axvline(365/7, linestyle='--', linewidth=2 , c='red')
    f, P = scipy.signal.periodogram(B.fluc, B.T, nfft=2**12)
    plt.plot(f, P,c='purple',linewidth=3,label='min',alpha=0.3)
    print('min')
    print('Influx:',np.sum(B.enter_SCI)/np.sum(A))
    print('Outflux:',np.sum(B.out_SCI)/np.sum(A))
    print('Intra-flux:',np.sum(B.In_SCI)/np.sum(A))

    if X!=None and X!=1:
        plt.xlabel(X,fontdict={'size': 40},labelpad=20)
    if Y!=None and X!=None:
        plt.ylabel(Y,fontdict={'size': 40})
    plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)
    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # ax.patch.set_alpha(0)
    if X==None:
        plt.yticks([])
        plt.xticks([])
    else:
        plt.yticks([])
        # ax.set_xticks([0,365/14,365/7,365/5,365/4,365/3,365/2])
        ax.set_xticks([0,365/7,365/4,365/3,365/2])
        ax.set_xticklabels(X_TICK_LABEL, fontsize=45)
    plt.tick_params(width=3,  length=5, pad=6, labelsize=45)

    ax1 = plt.gca()# 위치 지정 axes 생성

    ax1.yaxis.set_ticks([])          # 눈금 숫자(레이블) 없애기
    ax1.yaxis.set_ticklabels([])

    ax=plt.twinx()
    A=B.beta_m1_in+B.beta_m1_out+B.beta_m1_inter
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    ax.plot(f, P,c='olive',linewidth=2,label=r'$\beta=-1$',alpha=0.3)
    ax.yaxis.set_visible(False) 
    
    print('harmonic mean')
    print('Influx:',np.sum(B.beta_m1_in)/np.sum(A))
    print('Outflux:',np.sum(B.beta_m1_out)/np.sum(A))
    print('Intra-flux:',np.sum(B.beta_m1_inter)/np.sum(A))




    ax=plt.twinx()
    A=B.beta_1_in+B.beta_1_out+B.beta_1_inter
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    ax.plot(f, P,c='pink',linewidth=2,label=r'$\beta=1$',alpha=0.3)
    ax.yaxis.set_visible(False) 
    print('arithmetic mean')
    print('Influx:',np.sum(B.beta_1_in)/np.sum(A))
    print('Outflux:',np.sum(B.beta_1_out)/np.sum(A))
    print('Intra-flux:',np.sum(B.beta_1_inter)/np.sum(A))

    ax=plt.twinx()
    A=B.beta_0_in+B.beta_0_out+B.beta_0_inter
    f, P = scipy.signal.periodogram(A, B.T, nfft=2**12)
    ax.plot(f, P,c='brown',linewidth=2,label=r'$\sqrt{r_{t}r_{t-1}}$',alpha=0.3)
    ax.yaxis.set_visible(False) 
    print('geometric mean')
    print('Influx:',np.sum(B.beta_0_in)/np.sum(A))
    print('Outflux:',np.sum(B.beta_0_out)/np.sum(A))
    print('Intra-flux:',np.sum(B.beta_0_inter)/np.sum(A))
    # marker_set(locate,B)\    
    if locate==[0.74, 0.59, 0.2, 0.32]:
        ax=plt.twinx()
        ax.plot([],[],  c='purple',linewidth=3,label='min')#, marker='D', markerfacecolor='purple', markeredgecolor='black',markersize=m_size)
        ax.plot([],[],  c='brown',linewidth=3,label='geometric mean')#, marker='P', markerfacecolor='brown', markeredgecolor='black',markersize=m_size)
        # ax.plot([],[],  c='red',linewidth=2,label=r'$\beta=0$', marker='p', markerfacecolor='red', markeredgecolor='black',markersize=m_size)
        ax.plot([],[],  c='pink',linewidth=3,label='arithmetic mean')#, marker='X', markerfacecolor='pink', markeredgecolor='black',markersize=m_size)
        ax.plot([],[],  c='olive',linewidth=3,label='harmonic mean')#, marker='o', markerfacecolor='olive', markeredgecolor='black',markersize=m_size)
        plt.legend(fontsize=25)
        ax.yaxis.set_visible(False) 





# title_font = {    'fontsize': 30}
fontsizes=25
x=np.arange(0.01,10,0.1)
s=30

plt.figure(figsize=(35,19))
#plot (Netflix)
print('Netflix')
SI_beta([0.05, 0.59, 0.2, 0.32],netflix,'fluc',1,'Power Spectrum',None)
label_setting([0.03, 0.919, 0.01, 0.01],'A.Netflix',s)  

#plot (Hbo)
print('Hbo')
SI_beta([0.28, 0.59, 0.2, 0.32],Hbo,'fluc',1,None,None)
label_setting([0.26, 0.919, 0.01, 0.01],'B.HBO',s)

#plot (Disney)
print('NetDisneyflix')
SI_beta([0.51, 0.59, 0.2, 0.32],Disney,'fluc',1,None,None)
label_setting([0.49, 0.919, 0.01, 0.01],'C.Disney+',s) 

#plot (Amazon)
print('Amazon')
SI_beta([0.74, 0.59, 0.2, 0.32],prime,'fluc',1,None,'aggregate')
label_setting([0.72, 0.919, 0.01, 0.01],'D.Amazon Prime',s)  

#plot (UK)
print('Tw')
SI_beta([0.05, 0.19, 0.2, 0.32],tw_spotify,'fluc',X_label,'Power Spectrum',None)
label_setting([0.03, 0.519, 0.01, 0.01],'E.Spotify Taiwan',s) 

#plot (USA)
print('UK')
SI_beta([0.28, 0.19, 0.2, 0.32],gb_spotify,'fluc',X_label,None,None)
label_setting([0.26, 0.519, 0.01, 0.01],'F.Spotify UK',s)

#plot (TW)
print('USA')
SI_beta([0.51, 0.19, 0.2, 0.32],us_spotify,'fluc',X_label,None,None)
label_setting([0.49, 0.519, 0.01, 0.01],'G.Spotify USA',s)

#plot (au)
print('au')
SI_beta([0.74, 0.19, 0.2, 0.32],au_spotify,'fluc',X_label,None,None)
label_setting([0.72, 0.519, 0.01, 0.01],'H.Spotify Australia',s)  


save_fig('SI_beta_fig')
plt.close()
print('SI_beta_fig')
