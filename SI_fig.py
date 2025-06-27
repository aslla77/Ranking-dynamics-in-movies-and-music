
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pycountry
import pickle

max_PDF=1
first_PDF=1
last_PDF=1

N_b_c=1
N_b_fre=1
alpha_fre=1
co=1
last_first_PDF=1

xlabelpad=20
xtick_size=35

load_data='./data/'
save_result='./result/SI/'
save_data='./figure/SI/'
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
        selfinside_SCI=np.load(load_data+name+'_Interflux.npy')
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

        self.matrix=np.load(load_data+name+'_matrix.npy')
        self.Nlast=np.load(load_data+name+'_Nlast.npy')
        self.rb=np.load(load_data+name+'_rb.npy')
        self.rf=np.load(load_data+name+'_rf.npy')
        self.fx=np.load(load_data+name+'_fx.npy')
        self.prob=np.load(load_data+name+'_prob.npy')
        self.prob_1=np.load(load_data+name+'_prob1.npy')

        if not(name in ['netflix','Disney','Hbo','prime']):
            self.dsci=np.load(load_data+name+'_dSCI.npy')
            self.dr=np.load(load_data+name+'_dr.npy')
            self.ds=np.load(load_data+name+'_ds.npy')
            self.s_list=np.load(load_data+name+'_ds_list.npy')
            self.r_list=np.load(load_data+name+'_dr_list.npy')
        else:
            self.dr=np.load(load_data+name+'_dr.npy')


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
sw_spotify=setting('sw_spotify')
jp_spotify=setting('jp_spotify')



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

if N_b_c:
    for i in range(8):
        B=data_list[i]
        print(data_name[i])
        title_font = {    'fontsize': 30}

        x=np.arange(0.01,10,0.1)
        #20,25/_1: 22,23/_1: 24,24
        plt.figure(figsize=(24,24))
        s=30

        #aplot (원본)
        A=[0.05, 0.775, 0.85, 0.205]
        # N_b_change(A,B,name,s_date,e_date,X,Y,ytick=None,xtick=None):
        N_b=B.N_0
        alpha=N_b+1
        N_b_change(A,B,data_name[i],N_b,0,int(B.T/2),None,True,1,None)
        in_per,out_per,intra_per=per_check(B,N_b,B.N_0+1,data_name[i])
        label_setting([0.01, 0.989, 0.01, 0.01],f'$N_b$={N_b}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}',s)   # A=[0.05, 0.55, 0.2, 0.32])
        print(f'$N_b$={N_b}, $N_b+alpha$={alpha}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}')


        #bplot (0.25)
        A=[0.05, 0.525, 0.85, 0.205]
        N_b=int(B.N_0*0.75)
        N_b_change(A,B,data_name[i],N_b,0,int(B.T/2),None,True,1,None)
        in_per,out_per,intra_per=per_check(B,N_b,B.N_0+1,data_name[i])
        label_setting([0.01, 0.739, 0.01, 0.01],f'$N_b$={N_b}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}',s)   # A=[0.51, 0.55, 0.2, 0.32])
        print(f'$N_b$={N_b}, $N_b+alpha$={alpha}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}')


        #c (0.5)
        A=[0.05, 0.275, 0.85, 0.205]
        N_b=int(B.N_0*0.5)
        N_b_change(A,B,data_name[i],N_b,0,int(B.T/2),None,True,1,None)
        in_per,out_per,intra_per=per_check(B,N_b,B.N_0+1,data_name[i])
        label_setting([0.01, 0.489, 0.01, 0.01],f'$N_b$={N_b}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}',s)   # A=[0.05, 0.1, 0.2, 0.32])
        print(f'$N_b$={N_b}, $N_b+alpha$={alpha}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}')

        #d (0.75)
        A=[0.05, 0.025, 0.85, 0.205]
        N_b=int(B.N_0*0.25)
        N_b_change(A,B,data_name[i],N_b,0,int(B.T/2),True,True)
        in_per,out_per,intra_per=per_check(B,N_b,B.N_0+1,data_name[i])
        label_setting([0.01, 0.239, 0.01, 0.01],f'$N_b$={N_b}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}',s)   # A=[0.51, 0.1, 0.2, 0.32])
        print(f'$N_b$={N_b}, $N_b+alpha$={alpha}, influx={in_per:.3f}, outflux={out_per:.3f}, intraflux={intra_per:.3f}')


        save_fig(f'SI_1_N_b_{data_name[i]}')
        plt.close()
        print(f'SI_1_N_b_{data_name[i]}')

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
    alpha=B.N_0+1
    plt.axvline(365/7, linestyle='--', linewidth=2 , c='red',alpha=0.3)
    count=0
    for i in r:   
        N_b=int(B.N_0*i)
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


