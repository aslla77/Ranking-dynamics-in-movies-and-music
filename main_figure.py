
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pycountry
import pickle

xlabelpad=20
xtick_size=35

load_data='./data/'
save_result='./result/'
save_data='./figure/'
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

def persent_fluc(A):
  print(np.sum(A.fluc)/np.sum(A.In_SCI+A.out_SCI+A.enter_SCI))
  print('Influx(%):',np.sum(A.enter_SCI)/np.sum(A.In_SCI+A.out_SCI+A.enter_SCI))
  print('outflux(%):',np.sum(A.out_SCI)/np.sum(A.In_SCI+A.out_SCI+A.enter_SCI))
  print('intraflux(%):',np.sum(A.In_SCI)/np.sum(A.In_SCI+A.out_SCI+A.enter_SCI))

print("netflix")  
persent_fluc(netflix)
print("Disney")  
persent_fluc(Disney)
print("Hbo")  
persent_fluc(Hbo)
print("prime")  
persent_fluc(prime)
print("gb_spotify")  
persent_fluc(gb_spotify)
print("us_spotify")  
persent_fluc(us_spotify)
print("tw_spotify")  
persent_fluc(tw_spotify)
print("au_spotify")  
persent_fluc(au_spotify)

print('tw',tw_spotify.fluc)
print('gb',gb_spotify.fluc)
print('us',us_spotify.fluc)
print('au',au_spotify.fluc)

import pandas as pd
def per(data,N,check=False):
  max_i=max(data.fluc)
  min_i=min(data.fluc)

  frame=pd.DataFrame({'tot':data.fluc,
                      'Influx':data.enter_SCI,
                      'Outflux':data.out_SCI,
                      'inter':data.In_SCI})
  frame=frame.sort_values('tot',ascending=False).reset_index(drop=True)
  dt=data.T/N
  influx=np.zeros(N)
  outflux=np.zeros(N)
  inside=np.zeros(N)
  totfluc=np.zeros(N)

  for t in range(data.T):
    index= int(N-t//dt-1)
    if frame['tot'][t]==0:
      pass
    totfluc[int(index)]+=frame['tot'][t]
    influx[index]+=frame['Influx'][t]
    outflux[index]+=frame['Outflux'][t]
    inside[index]+=frame['inter'][t]
  influx/=totfluc
  outflux/=totfluc
  inside/=totfluc

  # plt.bar(np.arange(N),totfluc)
  if check :
    p1 = plt.bar(np.arange(N),influx, color='green')

    p2 = plt.bar(np.arange(N), inside, color='dodgerblue',
                bottom=influx) # stacked bar chart

    p3 = plt.bar(np.arange(N),outflux, color='orange',
                bottom=influx+inside)

    plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # plt.title('G. Spotify Taiwan',loc='left',fontdict=title_font)
    plt.xlabel('$top_sci$',labelpad=xlabelpad,fontdict={'weight': 'bold',
            'size': 40})
    plt.ylabel('$\%$',fontdict={'weight': 'bold',
            'size': 40})
  return totfluc,influx,outflux,inside

def per_1(data,N,check=False):
  max_i=max(data.fluc)
  min_i=min(data.fluc)

  frame=pd.DataFrame({'tot':data.fluc,
                      'up':data.up,
                      'down':data.down})
  frame=frame.sort_values('tot',ascending=False).reset_index(drop=True)
  dt=data.T/N
  up=np.zeros(N)
  down=np.zeros(N)
  totfluc=np.zeros(N)

  for t in range(data.T):
    index= int(N-t//dt-1)
    if frame['tot'][t]==0:
      pass
    totfluc[int(index)]+=frame['tot'][t]
    up[index]+=frame['up'][t]
    down[index]+=frame['down'][t]
  up/=totfluc
  down/=totfluc

  # plt.bar(np.arange(N),totfluc)
  if check :
    p1 = plt.bar(np.arange(N),up, color='green')

    p2 = plt.bar(np.arange(N), down, color='dodgerblue',
                bottom=up) # stacked bar chart


    plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # plt.title('G. Spotify Taiwan',loc='left',fontdict=title_font)
    plt.xlabel('$top_sci$',labelpad=xlabelpad,fontdict={'weight': 'bold',
            'size': 40})
    plt.ylabel('$\%$',fontdict={'weight': 'bold',
            'size': 40})
    plt.show
  return totfluc,up,down

def day_dis(data,start_day,T):
  # 2. 날짜 생성 (2023-01-01부터 시작)
  dates = pd.date_range(start=start_day, periods=T, freq='D')

  # 3. 상위 10% 값 계산
  threshold = np.percentile(data, 90)  # 상위 10% 기준값

  # 4. 상위 10%에 해당하는 데이터의 날짜 찾기
  top_10_percent_dates = dates[data >= threshold]

  # 5. 해당 날짜들의 요일 추출
  top_10_percent_weekdays = top_10_percent_dates.day_name()

  # 6. 요일 분포 확인
  weekday_counts = top_10_percent_weekdays.value_counts()

  return top_10_percent_weekdays,weekday_counts

def day_dis20(data,start_day,T):
  # 2. 날짜 생성 (2023-01-01부터 시작)
  dates = pd.date_range(start=start_day, periods=T, freq='D')

  # 3. 상위 10% 값 계산
  threshold = np.percentile(data, 80)  # 상위 10% 기준값

  # 4. 상위 10%에 해당하는 데이터의 날짜 찾기
  top_10_percent_dates = dates[data >= threshold]

  # 5. 해당 날짜들의 요일 추출
  top_10_percent_weekdays = top_10_percent_dates.day_name()

  # 6. 요일 분포 확인
  weekday_counts = top_10_percent_weekdays.value_counts()

  return top_10_percent_weekdays,weekday_counts

def day_PDF(data,start_day,T):
  # 예시 데이터
  date_range = pd.date_range(start=start_day,  periods=T, freq='D')

  # values = np.random.rand(len(date_range))  # 임의 값

  # DataFrame 만들기
  df = pd.DataFrame({'Date': date_range, 'Value': data})
  df['Week'] = df['Date'].dt.isocalendar().week
  df['Year'] = df['Date'].dt.isocalendar().year
  df['DayOfWeek'] = df['Date'].dt.day_name()

  # 각 주마다 top1, top2 요일 저장
  top1_days = []
  top2_days = []

  for (year, week), group in df.groupby(['Year', 'Week']):
      if len(group) < 2:
          continue
      top2 = group.nlargest(2, 'Value')
      top1_days.append(top2.iloc[0]['DayOfWeek'])
      top2_days.append(top2.iloc[1]['DayOfWeek'])

  # 요일 순서 정의
  day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  # 빈도 계산 및 PDF로 변환
  top1_counts = pd.Series(top1_days).value_counts().reindex(day_order, fill_value=0)
  top2_counts = pd.Series(top2_days).value_counts().reindex(day_order, fill_value=0)

  top1_pdf = top1_counts / top1_counts.sum()
  top2_pdf = top2_counts / top2_counts.sum()
  # print(1,np.sum(top1_pdf))
  # print(2,np.sum(top2_pdf))
  return top1_pdf, top2_pdf

def day_PDF_S(data,start_day,T):
  # 예시 데이터
  date_range = pd.date_range(start=start_day,  periods=T, freq='D')

  # values = np.random.rand(len(date_range))  # 임의 값

  # DataFrame 만들기
  df = pd.DataFrame({'Date': date_range, 'Value': data})
  df['Week'] = df['Date'].dt.to_period('W-SAT')
  df['DayOfWeek'] = df['Date'].dt.day_name()

  # 각 주마다 top1, top2 요일 저장
  top1_days = []
  top2_days = []

  for week, group in df.groupby([ 'Week']):
      if len(group) < 2:
          continue
      top2 = group.nlargest(2, 'Value')
      top1_days.append(top2.iloc[0]['DayOfWeek'])
      top2_days.append(top2.iloc[1]['DayOfWeek'])

  # 요일 순서 정의
  day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  # 빈도 계산 및 PDF로 변환
  top1_counts = pd.Series(top1_days).value_counts().reindex(day_order, fill_value=0)
  top2_counts = pd.Series(top2_days).value_counts().reindex(day_order, fill_value=0)

  top1_pdf = top1_counts / top1_counts.sum()
  top2_pdf = top2_counts / top2_counts.sum()
  print(1,np.sum(top1_pdf))
  print(2,np.sum(top2_pdf))
  return top1_pdf, top2_pdf

def day_sum(data,start_day,T):
  day_index={'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
  date_range = pd.date_range(start=start_day,  periods=T, freq='D')
  result=np.zeros(7)
  for i in range(T):
    dt = pd.to_datetime(date_range[i])
    index=dt.day_name()
    result[day_index[index]]+=data[i]

  return result

def rank_move(A):
  T20=np.zeros(2*(A.N_0))
  M20=np.zeros(2*(A.N_0))
  B20=np.zeros(2*(A.N_0))
  B=np.sum(A.matrix,axis=0)
  t=int(round((A.N_0+1)*0.8))
  b=A.N_0
  for index in range(t,b):
    for jndex in range(len(B[index])-1):
      B20[A.N_0+jndex-index]+=B[index,jndex]
  t=int(round((A.N_0+1)*0.4))
  b=int(round((A.N_0+1)*0.6))
  for index in range(t,b):
    for jndex in range(len(B[index])-1):
      M20[A.N_0+jndex-index]+=B[index,jndex]
  t=0
  b=int(round((A.N_0+1)*0.2))
  for index in range(t,b):
    for jndex in range(len(B[index])-1):
      T20[A.N_0+jndex-index]+=B[index,jndex]

  T20=T20/np.sum(T20)
  M20=M20/np.sum(M20)
  B20=B20/np.sum(B20)
  x=np.arange(-(A.N_0),(A.N_0))
  return T20,M20,B20,x

def per_1(data,N,check=False):
  max_i=max(data.fluc)
  min_i=min(data.fluc)

  frame=pd.DataFrame({'tot':data.fluc,
                      'up':data.up,
                      'down':data.down})
  frame=frame.sort_values('tot',ascending=False).reset_index(drop=True)
  dt=data.T/N
  up=np.zeros(N)
  down=np.zeros(N)
  totfluc=np.zeros(N)

  for t in range(data.T):
    index= int(N-t//dt-1)
    # print(t,frame['tot'][t],index,frame['Influx'][t])
    if frame['tot'][t]==0:
      pass
    totfluc[int(index)]+=frame['tot'][t]
    up[index]+=frame['up'][t]
    down[index]+=frame['down'][t]
  # print(totfluc)
  up/=totfluc
  down/=totfluc

  # plt.bar(np.arange(N),totfluc)
  if check :
    p1 = plt.bar(np.arange(N),up, color='green')

    p2 = plt.bar(np.arange(N), down, color='dodgerblue',
                bottom=up) # stacked bar chart


    plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

    ax=plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # plt.title('G. Spotify Taiwan',loc='left',fontdict=title_font)
    plt.xlabel('$top_sci$',labelpad=xlabelpad,fontdict={'weight': 'bold',
            'size': 40})
    plt.ylabel('$\%$',labelpad=xlabelpad,fontdict={'weight': 'bold',
            'size': 40})
    plt.show
  return totfluc,up,down

def save_fig(A):
  plt.savefig(save_data+A+'.png',bbox_inches='tight')
  plt.savefig(save_result+A+'.pdf',bbox_inches='tight')    
  # plt.savefig(save_data+A+'.png',bbox_inches='tight',transparent=True)
  # plt.savefig(save_result+A+'.pdf',bbox_inches='tight',transparent=True)    




# # # # ======================================================================================================================================================================


def fig_2_set(A,B,s_date,e_date,X,Y,ytick=None,xtick=None):
  plt.axes(A)
  c=False
  if s_date==None and e_date==None:
    arang=B.T
    p1 = plt.bar(np.arange(arang), B.enter_SCI, color='green')

    p2 = plt.bar(np.arange(arang), B.In_SCI, color='dodgerblue',
                bottom=B.enter_SCI) # stacked bar chart

    p3 = plt.bar(np.arange(arang),B.out_SCI, color='orange',
                bottom=B.enter_SCI+B.In_SCI)

  else:
    c=True
    arang=e_date-s_date
    p1 = plt.bar(np.arange(arang), B.enter_SCI[s_date:e_date], color='green')

    p2 = plt.bar(np.arange(arang), B.In_SCI[s_date:e_date], color='dodgerblue',
                bottom=B.enter_SCI[s_date:e_date]) # stacked bar chart

    p3 = plt.bar(np.arange(arang),B.out_SCI[s_date:e_date], color='orange',
                bottom=B.enter_SCI[s_date:e_date]+B.In_SCI[s_date:e_date])
  
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
  if xtick!=None:
    plt.xticks([])
  else:
    ax=plt.gca()
    ax.set_xticks(np.arange(0,arang,30))
  plt.tick_params(width=3,  length=5, pad=6, labelsize=50)
  
  if c and arang>=int(B.T/2):
    plt.xlim(0,arang)
  else:
    plt.xlim(0,arang)

# #@title it_barplot
title_font = {    'fontsize': 30}

x=np.arange(0.01,10,0.1)

plt.figure(figsize=(30,20))
#aplot (Netflix)
A=[0.05, 0.88, 0.95, 0.1]
fig_2_set(A,netflix,None,None,None,True,1,1)

#bplot (Disney)
A=[0.05, 0.755, 0.95, 0.1]
fig_2_set(A,Hbo,None,None,None,True,1,1)

#cplot (Disney+)
A=[0.05, 0.63, 0.95, 0.1]
fig_2_set(A,Disney,None,None,None,True,1,1)

#dplot (prime)
A=[0.05, 0.505, 0.95, 0.1]
fig_2_set(A,prime,None,None,None,True,1,1)

#E
A=[0.05, 0.38, 0.95, 0.1]
fig_2_set(A,gb_spotify,None,None,None,True,1,1)

#F
A=[0.05, 0.255, 0.95, 0.1]
fig_2_set(A,us_spotify,None,None,None,True,1,1)

#G
A=[0.05, 0.13, 0.95, 0.1]
fig_2_set(A,tw_spotify,None,None,None,True,1,1)

#H
A=[0.05, 0.005, 0.95, 0.1]
fig_2_set(A,au_spotify,None,None,True,True,1)


s=15
label_setting([0.01, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
label_setting([0.01, 0.864, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
label_setting([0.01, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
label_setting([0.01, 0.614, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
label_setting([0.01, 0.489, 0.01, 0.01],'E. Spotify UK',s)   # A=[0.05, 0.1, 0.2, 0.32])
label_setting([0.01, 0.365, 0.01, 0.01],'F. Spotify USA',s)   # A=[0.28, 0.1, 0.2, 0.32])
label_setting([0.01, 0.239, 0.01, 0.01],'G. Spotify Taiwan',s)   # A=[0.51, 0.1, 0.2, 0.32])
label_setting([0.01, 0.114, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

save_fig('fig_2_all')
plt.close()
print('fig_2_all')

# # # # ======================================================================================================================================================================

# #@title it_barplot
title_font = {    'fontsize': 30}

x=np.arange(0.01,10,0.1)

plt.figure(figsize=(30,25))
#aplot (Netflix)
A=[0.05, 0.885, 0.95, 0.1]
fig_2_set(A,netflix,0,int(netflix.T/2),None,True,1,1)

#bplot (Disney)
A=[0.05, 0.76, 0.95, 0.1]
fig_2_set(A,Hbo,0,int(Hbo.T/2),None,True,1,1)

#cplot (Disney+)
A=[0.05, 0.635, 0.95, 0.1]
fig_2_set(A,Disney,0,int(Disney.T/2),None,True,1,1)

#dplot (prime)
A=[0.05, 0.51, 0.95, 0.1]
fig_2_set(A,prime,0,int(prime.T/2),None,True,1,1)

#E
A=[0.05, 0.385, 0.95, 0.1]
fig_2_set(A,tw_spotify,0,int(tw_spotify.T/2),None,True,1,1)

#F
A=[0.05, 0.26, 0.95, 0.1]
fig_2_set(A,gb_spotify,0,int(gb_spotify.T/2),None,True,1,1)

#G
A=[0.05, 0.135, 0.95, 0.1]
fig_2_set(A,us_spotify,0,int(us_spotify.T/2),None,True,1,1)

#H
A=[0.05, 0.01, 0.95, 0.1]
fig_2_set(A,au_spotify,0,int(au_spotify.T/2),True,True,1)


s=35
label_setting([0.03, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
label_setting([0.03, 0.864, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
label_setting([0.03, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
label_setting([0.03, 0.614, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
label_setting([0.03, 0.489, 0.01, 0.01],'E. Spotify Taiwan',s)   # A=[0.05, 0.1, 0.2, 0.32])
label_setting([0.03, 0.365, 0.01, 0.01],'F. Spotify UK',s)   # A=[0.28, 0.1, 0.2, 0.32])
label_setting([0.03, 0.239, 0.01, 0.01],'G. Spotify USA',s)   # A=[0.51, 0.1, 0.2, 0.32])
label_setting([0.03, 0.114, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

save_fig('fig_2_half')
plt.close()
print('fig_2_half')

# # # # ======================================================================================================================================================================

#@title it_barplot
title_font = {    'fontsize': 30}
label_s=15
N=10
centers = np.arange(0,1,1/N)
x=np.arange(0.01,10,0.1)

plt.figure(figsize=(30,15))
#aplot (Netflix)
plt.axes([0.05, 0.55, 0.2, 0.32])
totfluc,influx,outflux,inside=per(netflix,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)


plt.ylabel(r'$g^{\alpha}_j$',fontdict={'size': 40},loc='center',labelpad=10)

# plt.title('A. Netflix',loc='left',fontdict=title_font)

#bplot (Hbo)
plt.axes([0.28, 0.55, 0.2, 0.32])

totfluc,influx,outflux,inside=per(Hbo,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)



# plt.title('B. HBO',loc='left',fontdict=title_font)

#cplot (Disney+)
plt.axes([0.51, 0.55, 0.2, 0.32])
totfluc,influx,outflux,inside=per(Disney,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)


# plt.title('C. Disney+',loc='left',fontdict=title_font)

#dplot (prime)
plt.axes([0.74, 0.55, 0.2, 0.32])
totfluc,influx,outflux,inside=per(prime,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)


# plt.title('D. Amazon Prime',loc='left',fontdict=title_font)

#E
plt.axes([0.05, 0.1, 0.2, 0.32])
totfluc,influx,outflux,inside=per(tw_spotify,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=25)
for spine in ax.spines.values():
    spine.set_linewidth(3)

# plt.title('E. Spotify Sweden',loc='left',fontdict=title_font)
plt.ylabel(r'$g^{\alpha}_j$',fontdict={'size': 40},loc='center',labelpad=10)
plt.xlabel('Group index $j$',labelpad=xlabelpad,fontdict={         'size': 40})
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)

#F
plt.axes([0.28, 0.1, 0.2, 0.32])

totfluc,influx,outflux,inside=per(gb_spotify,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)
plt.xlabel('Group index $j$',labelpad=xlabelpad,fontdict={         'size': 40})
# plt.title('F. Spotify USA',loc='left',fontdict=title_font)

#G
plt.axes([0.51, 0.1, 0.2, 0.32])
totfluc,influx,outflux,inside=per(us_spotify,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)
plt.xlabel('Group index $j$',labelpad=xlabelpad,fontdict={         'size': 40})
# plt.title('G. Spotify Taiwan',loc='left',fontdict=title_font)

#H
plt.axes([0.74, 0.1, 0.2, 0.32])
totfluc,influx,outflux,inside=per(au_spotify,10)
p1 = plt.bar(centers,influx[::-1],  width=0.07,color='green')

p2 = plt.bar(centers, inside[::-1], width=0.07, color='dodgerblue',
            bottom=influx[::-1]) # stacked bar chart

p3 = plt.bar(centers,outflux[::-1], width=0.07, color='orange',
            bottom=influx[::-1]+inside[::-1])

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=25)
ax=plt.gca()
ax.set_xticks([0,0.4,0.9])
ax.set_xticklabels([1,5,10], fontsize=35)
for spine in ax.spines.values():
    spine.set_linewidth(3)
plt.xlabel('Group index $j$',labelpad=xlabelpad,fontdict={         'size': 40})

# plt.title('H. Spotify Japan',loc='left',fontdict=title_font)

s=30
label_setting([0.03, 0.89, 0.01, 0.01],'A. Netflix',s)   # plt.axes([0.05, 0.55, 0.2, 0.32])
label_setting([0.26, 0.89, 0.01, 0.01],'B. HBO',s)   # plt.axes([0.28, 0.55, 0.2, 0.32])
label_setting([0.49, 0.89, 0.01, 0.01],'C. Disney+',s)   # plt.axes([0.51, 0.55, 0.2, 0.32])
label_setting([0.72, 0.89, 0.01, 0.01],'D. Amazon Prime',s)    # plt.axes([0.74, 0.55, 0.2, 0.32])
label_setting([0.03, 0.44, 0.01, 0.01],'E. Spotify Taiwan',s)   # plt.axes([0.05, 0.1, 0.2, 0.32])
label_setting([0.26, 0.44, 0.01, 0.01],'F. Spotify UK',s)   # plt.axes([0.28, 0.1, 0.2, 0.32])
label_setting([0.49, 0.44, 0.01, 0.01],'G. Spotify USA',s)   # plt.axes([0.51, 0.1, 0.2, 0.32])
label_setting([0.72, 0.44, 0.01, 0.01],'H. Spotify Australia',s)   # plt.axes([0.74, 0.1, 0.2, 0.32])
save_fig('fig_3')
plt.close()
print('fig_3')
# # # # ======================================================================================================================================================================

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
  if X==None:
    plt.yticks([])
    plt.xticks([])
  else:
    plt.yticks([])
    # ax.set_xticks([0,365/14,365/7,365/5,365/4,365/3,365/2])
    ax.set_xticks([0,365/7,365/4,365/3,365/2])
    ax.set_xticklabels(X_TICK_LABEL, fontsize=25)
  plt.tick_params(width=3,  length=5, pad=6, labelsize=35)

def fig_4_set(A,B,C,X,Y,label1):
  plt.axes(A)
  plt.axvline(365/7, linestyle='--', linewidth=2 , c='red')
  if C=='fluc':
    f, P = scipy.signal.periodogram(B.fluc, B.T, nfft=2**12)
    # plt.plot(f, P,c='#536512',linewidth=2,label=label1)
    plt.plot(f, P,c='purple',linewidth=2,label=label1)
  else:
    f, P = scipy.signal.periodogram(B.F_list, B.T, nfft=2**12)
    plt.plot(f, P,c='#88C273',linewidth=2,label=label1)
  # if label1!=None:
  #   plt.legend(fontsize = 25,loc = 'upper right')

  if X!=None and X!=1:
    plt.xlabel(X,fontdict={'size': 40},labelpad=20)
  if Y!=None and X!=None:
    plt.ylabel(Y,fontdict={'size': 40})
  fig_4_tick(X)

# title_font = {    'fontsize': 30}
fontsizes=25
x=np.arange(0.01,10,0.1)
s=30

plt.figure(figsize=(35,19))
#plot (Netflix)
fig_4_set([0.05, 0.59, 0.2, 0.32],netflix,'fluc',1,'Power Spectrum',None)
label_setting([0.03, 0.919, 0.01, 0.01],'A.Netflix',s)  

#plot (Hbo)
fig_4_set([0.28, 0.59, 0.2, 0.32],Hbo,'fluc',1,None,None)
label_setting([0.26, 0.919, 0.01, 0.01],'B.HBO',s)

#plot (Disney)
fig_4_set([0.51, 0.59, 0.2, 0.32],Disney,'fluc',1,None,None)
label_setting([0.49, 0.919, 0.01, 0.01],'C.Disney+',s) 

#plot (Amazon)
fig_4_set([0.74, 0.59, 0.2, 0.32],prime,'fluc',1,None,'aggregate')
label_setting([0.72, 0.919, 0.01, 0.01],'D.Amazon Prime',s)  

#plot (UK)
fig_4_set([0.05, 0.19, 0.2, 0.32],tw_spotify,'fluc',X_label,'Power Spectrum',None)

label_setting([0.03, 0.519, 0.01, 0.01],'E.Spotify Taiwan',s) 

#plot (USA)
fig_4_set([0.28, 0.19, 0.2, 0.32],gb_spotify,'fluc',X_label,None,None)

label_setting([0.26, 0.519, 0.01, 0.01],'F.Spotify UK',s)

#plot (TW)
fig_4_set([0.51, 0.19, 0.2, 0.32],us_spotify,'fluc',X_label,None,None)

label_setting([0.49, 0.519, 0.01, 0.01],'G.Spotify USA',s)

#plot (au)
fig_4_set([0.74, 0.19, 0.2, 0.32],au_spotify,'fluc',X_label,None,None)
label_setting([0.72, 0.519, 0.01, 0.01],'H.Spotify Australia',s)  


save_fig('fig_4')
plt.close()
print('fig_4')

# # # ======================================================================================================================================================================
xtick_size=35
plt.figure(figsize=(30,15))
weekday_order_abbr = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#aplot (Netflix)
plt.axes([0.05, 0.55, 0.2, 0.32])
top_10_percent_weekdays,weekday_counts=day_dis(netflix.fluc,ott_start, netflix.T)

print(top_10_percent_weekdays)
print(weekday_counts)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,21,5))
plt.xlabel('')
plt.ylabel('Count',fontdict={         'size': 40},loc='center',labelpad=15)
# plt.title('A. Netflix',loc='left',fontdict=title_font)

#bplot (Disney)
plt.axes([0.28, 0.55, 0.2, 0.32])

top_10_percent_weekdays,weekday_counts=day_dis(Hbo.fluc,ott_start, Hbo.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,16,5))
plt.ylabel('')
plt.xlabel('')
# plt.title('B. HBO',loc='left',fontdict=title_font)

#cplot (Disney+)
plt.axes([0.51, 0.55, 0.2, 0.32])

top_10_percent_weekdays,weekday_counts=day_dis(Disney.fluc,ott_start, Disney.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)

#그래프 정리
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,16,5))    
plt.xlabel('')
plt.ylabel('')
# plt.title('C. Disney+',loc='left',fontdict=title_font)

#dplot (prime)
plt.axes([0.74, 0.55, 0.2, 0.32])

top_10_percent_weekdays,weekday_counts=day_dis(prime.fluc,ott_start, prime.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


#그래프 정리
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)

ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,16,5))
plt.xlabel('')
plt.ylabel('')
# plt.title('D. Amazon Prime',loc='left',fontdict=title_font)

#E
plt.axes([0.05, 0.1, 0.2, 0.32])
top_10_percent_weekdays,weekday_counts=day_dis(tw_spotify.fluc,spotify_start, tw_spotify.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

#그래프 정리
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)
ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,16,5))
plt.ylabel('Count',fontdict={         'size': 40},loc='center',labelpad=15)
# plt.title('E. Spotify Sweden',loc='left',fontdict=title_font)
plt.xlabel('Day of the week',labelpad=xlabelpad,fontdict={         'size': 40})


#F
plt.axes([0.28, 0.1, 0.2, 0.32])

top_10_percent_weekdays,weekday_counts=day_dis(gb_spotify.fluc,spotify_start, gb_spotify.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

#그래프 정리
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)
ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,21,5))
# plt.title('F. Spotify USA',loc='left',fontdict=title_font)
plt.ylabel('')
plt.xlabel('Day of the week',labelpad=xlabelpad,fontdict={         'size': 40})

#G

plt.axes([0.51, 0.1, 0.2, 0.32])
top_10_percent_weekdays,weekday_counts=day_dis(us_spotify.fluc,spotify_start, us_spotify.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

#그래프 정리
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)
ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,21,5))
# plt.title('G. Spotify Taiwan',loc='left',fontdict=title_font)
plt.ylabel('')
plt.xlabel('Day of the week',labelpad=xlabelpad,fontdict={         'size': 40})

#H

plt.axes([0.74, 0.1, 0.2, 0.32])
top_10_percent_weekdays,weekday_counts=day_dis(au_spotify.fluc,spotify_start, au_spotify.T)
sns.countplot(x=top_10_percent_weekdays,orient='v', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


#그래프 정리
plt.xticks(ticks=range(7), labels=weekday_order_abbr, fontsize=30,rotation=45)
plt.tick_params(width=3,  length=5, pad=6, labelsize=xtick_size)
ax=plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.set_yticks(np.arange(0,11,5))
plt.ylabel('')
plt.xlabel('Day of the week',labelpad=xlabelpad,fontdict={         'size': 40})


s=30
label_setting([0.03, 0.89, 0.01, 0.01],'A. Netflix',s)   # plt.axes([0.05, 0.55, 0.2, 0.32])
label_setting([0.26, 0.89, 0.01, 0.01],'B. HBO',s)   # plt.axes([0.28, 0.55, 0.2, 0.32])
label_setting([0.49, 0.89, 0.01, 0.01],'C. Disney+',s)   # plt.axes([0.51, 0.55, 0.2, 0.32])
label_setting([0.72, 0.89, 0.01, 0.01],'D. Amazon Prime',s)    # plt.axes([0.74, 0.55, 0.2, 0.32])
label_setting([0.03, 0.44, 0.01, 0.01],'E. Spotify Taiwan',s)   # plt.axes([0.05, 0.1, 0.2, 0.32])
label_setting([0.26, 0.44, 0.01, 0.01],'F. Spotify UK',s)   # plt.axes([0.28, 0.1, 0.2, 0.32])
label_setting([0.49, 0.44, 0.01, 0.01],'G. Spotify USA',s)   # plt.axes([0.51, 0.1, 0.2, 0.32])
label_setting([0.72, 0.44, 0.01, 0.01],'H. Spotify Australia',s)   # plt.axes([0.74, 0.1, 0.2, 0.32])

save_fig('fig_5')
plt.close()
print('fig_5')


# # # ======================================================================================================================================================================

import boxp

with open(file='./data/netflix_Nb.pickle', mode='rb') as f:
    Netflix=pickle.load(f)

with open(file='./data/hbo_Nb.pickle', mode='rb') as f:
    Hbo=pickle.load(f)

with open(file='./data/disney_Nb.pickle', mode='rb') as f:
    Disney=pickle.load(f)

with open(file='./data/prime_Nb.pickle', mode='rb') as f:
    Prime=pickle.load(f)

with open(file='./data/tw_Nb.pickle', mode='rb') as f:
    Tw=pickle.load(f)

with open(file='./data/gb_Nb.pickle', mode='rb') as f:
    UK=pickle.load(f)

with open(file='./data/us_Nb.pickle', mode='rb') as f:
    USA=pickle.load(f)

with open(file='./data/au_Nb.pickle', mode='rb') as f:
    AU=pickle.load(f)


# def fig_6_set(A,B,s_date,e_date,X,Y,ytick=None,xtick=None):
def fig_6_set(A,C,X,Y,ytick=None,xtick=None,legen=False):
  plt.axes(A)
  boxp.boxp1(C,10,True,False,False,legen)
  
  ax=plt.gca()
  for spine in ax.spines.values():
      spine.set_linewidth(3)
  # plt.yticks([])
  if X!= None:
    plt.xlabel('Group index $j$',labelpad=xlabelpad,fontdict={         'size': 40})
  if Y!= None:
    plt.ylabel('Lifetime',fontdict={         'size': 40},loc='center',labelpad=15)
  if xtick!=None:
    plt.xticks([])
    
  plt.tick_params(width=3,  length=5, pad=6, labelsize=30)
  

# #@title it_barplot
title_font = {    'fontsize': 30}

x=np.arange(0.01,10,0.1)

plt.figure(figsize=(24,24))
#aplot (Netflix)
A=[0.05, 0.76, 0.425, 0.22]
fig_6_set(A,Netflix,None,True,1,1)

#bplot (Disney)
A=[0.55, 0.76, 0.425, 0.22]
fig_6_set(A,Hbo,None,None,1,1,1)

#cplot (Disney+)
A=[0.05, 0.51, 0.425, 0.22]
fig_6_set(A,Disney,None,True,1,1)

#dplot (prime)
A=[0.55, 0.51, 0.425, 0.22]
fig_6_set(A,Prime,None,None,1,1)

#E
A=[0.05, 0.26, 0.425, 0.22]
fig_6_set(A,Tw,None,True,1,1)

#F
A=[0.55, 0.26, 0.425, 0.22]
fig_6_set(A,UK,None,None,1,1)

#G
A=[0.05, 0.01, 0.425, 0.22]
fig_6_set(A,USA,True,True,1)

#H
A=[0.55, 0.01, 0.425, 0.22]
fig_6_set(A,AU,True,None,1)


s=30
label_setting([0.01, 0.989, 0.01, 0.01],'A. Netflix',s)   # A=[0.05, 0.55, 0.2, 0.32])
label_setting([0.51, 0.989, 0.01, 0.01],'B. HBO',s)   # A=[0.28, 0.55, 0.2, 0.32])
label_setting([0.01, 0.739, 0.01, 0.01],'C. Disney+',s)   # A=[0.51, 0.55, 0.2, 0.32])
label_setting([0.51, 0.739, 0.01, 0.01],'D. Amazon Prime',s)    # A=[0.74, 0.55, 0.2, 0.32])
label_setting([0.01, 0.489, 0.01, 0.01],'E. Spotify Taiwan',s)   # A=[0.05, 0.1, 0.2, 0.32])
label_setting([0.51, 0.489, 0.01, 0.01],'F. Spotify UK',s)   # A=[0.28, 0.1, 0.2, 0.32])
label_setting([0.01, 0.239, 0.01, 0.01],'G. Spotify USA',s)   # A=[0.51, 0.1, 0.2, 0.32])
label_setting([0.51, 0.239, 0.01, 0.01],'H. Spotify Australia',s)   # A=[0.74, 0.1, 0.2, 0.32])

save_fig('fig_6')
plt.close()
print('fig_6')
# # # ======================================================================================================================================================================

# def fig_6_set(A,B,s_date,e_date,X,Y,ytick=None,xtick=None):
def fig_7_set(A,C,x_label=False,y_label=False):
  plt.axes(A)
  boxp.binning2(C,20,True,False,False,x_label,y_label)
  
  ax=plt.gca()
  for spine in ax.spines.values():
      spine.set_linewidth(3)
  # plt.yticks([])
    
  plt.tick_params(width=3,  length=5, pad=6, labelsize=30)
  

# #@title it_barplot
title_font = {    'fontsize': 30}

x=np.arange(0.01,10,0.1)
#20,25/_1: 22,23/_1: 24,24
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

save_fig('fig_7')
plt.close()
print('fig_7')