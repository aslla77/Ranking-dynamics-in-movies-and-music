# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pycountry
import scipy.signal
from datetime import datetime, timedelta


class Analysis():
  def __init__(self,data):

    # Generate basic data (T,N_0,N_{T-1})
    self.data=data
    self.T=len(data['T'].unique())
    self.N_0=len(data['T'].loc[data['T']==data['T'].unique()[0]])
    self.N_last=len(data.Name.unique()) #N_{t-1}
    print('T:',self.T)
    print('N_0:',self.N_0)
    print('N_tot:',self.N_last)

    #array (o_t) and array(F_t)
    self.o_list=np.zeros(self.T)
    self.F_list=np.zeros(self.T)
    before_data=set(data[:self.N_0].Name)
    for i in range(self.T):
      self.o_list[i]=len(data[:self.N_0*(i+1)].Name.unique())/self.N_0
      check_data=set(data[self.N_0*(i):self.N_0*(i+1)].Name)
      self.F_list[i]=len(check_data-before_data)/self.N_0
      before_data=check_data.copy()

    # \dot o and <F>
    self.mean_F=sum(self.F_list)/(self.T-1)
    self.dot_o=(self.o_list[-1]-self.o_list[0])/(self.T-1)



  def measure(self, boundary=False):
    # create matrix
    if boundary == False:
        boundary = self.N_0 + 1
    self.over = []                              # Store time points when SCI exceeds a threshold
    self.matrix = np.zeros((self.T, self.N_0 + 1, self.N_0 + 1)).astype(float)  # Matrix

    self.fluc = np.zeros(self.T)                # SCI (signed)
    self.fluc_1 = np.zeros(self.T)              # SCI without absolute values
    self.move_rank = np.zeros([2, self.N_0])    # Entry positions [1], exit positions [0]
    self.enter_SCI = np.zeros(self.T)           # SCI due to entry
    self.out_SCI = np.zeros(self.T)             # SCI due to exit
    self.In_SCI = np.zeros(self.T)              # Internal SCI
    self.new_influx = np.zeros(self.T)          # SCI of new entries
    self.re_influx = np.zeros(self.T)           # SCI of re-entries
    self.new_o_list = np.zeros(self.T)          # Number of new entries
    self.up = np.zeros(self.T)                  # Count of upward movements
    self.down = np.zeros(self.T)                # Count of downward movements

    #create measur dict Name :
    # [0. Lifetime, 
    #  1. First rank, 
    #  2. Highest rank, 
    #  3. Last rank, 
    #  4. Average rank, 
    #  5. Rank variance, 
    #  6. Rank fluctuation timeline, 
    #  7. Contribution to SCI change by the item, 
    #  8. Same as 7 but without absolute values, 
    #  9. Number of entries, 
    # 10. Re-entry rank distribution]

    self.measure_data=dict(zip(self.data.Name.unique(),[[0,0,np.inf,0,0,[],[],[],[],0,[],None] for i in range(self.N_last)]))


    data_name=set(list(self.data.Name.unique()))
    self.mean_i=[]
    self.std_i=[]
    # Data Analysis

    # Data at t
    before_data=dict(zip(self.data[:self.N_0].Name,self.data[:self.N_0].Rank))

    for time in range(self.T):
      # Data at t+1
      check_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))

      for item in data_name:
        #아이템별 분석
        if item in list(check_data.keys()):
          
          self.measure_data[item][0]+=1

          if self.measure_data[item][1]==0:
            self.measure_data[item][1]=check_data[item]

          if self.measure_data[item][2]>=check_data[item]:
            self.measure_data[item][2]=check_data[item]

          self.measure_data[item][3]=check_data[item]
          self.measure_data[item][6].append(check_data[item])

        else:
          
          self.measure_data[item][6].append(np.nan)

        if self.measure_data[item][11]== None and (item in list(check_data.keys()) or item in list(before_data.keys())):
          self.measure_data[item][11]=time
          # print(item,':',time)

        if time!=0:
          if (item in list(check_data.keys())) and (item in list(before_data.keys())):
            self.matrix[time][before_data[item]-1][check_data[item]-1]+=1
            self.In_SCI[time]+=abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item])*self.N_0)

            self.fluc[time]+=abs(check_data[item]-before_data[item])/(min(check_data[item],before_data[item])*self.N_0)
            self.fluc_1[time]+=(before_data[item]-check_data[item])/(min(check_data[item],before_data[item])*self.N_0)

            self.measure_data[item][7].append(abs(check_data[item]-before_data[item])/(min(check_data[item],before_data[item])*self.N_0))
            self.measure_data[item][8].append((before_data[item]-check_data[item])/(min(check_data[item],before_data[item])*self.N_0))


            if before_data[item]-check_data[item]>0:
              self.up[time]+=abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item])*self.N_0)
            elif before_data[item]-check_data[item]<0:
              self.down[time]+=abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item])*self.N_0)

          #outflux
          elif not(item in list(check_data.keys())) and (item in list(before_data.keys())):
            self.matrix[time][before_data[item]-1][self.N_0]+=1
            self.out_SCI[time]+=abs(before_data[item]-(boundary))/(before_data[item]*self.N_0)
            self.move_rank[0][before_data[item]-1]+=1
            self.fluc[time]+=abs((boundary)-before_data[item])/(before_data[item]*self.N_0)
            self.fluc_1[time]+=(before_data[item]-(boundary))/(before_data[item]*self.N_0)
            self.down[time]+=abs((boundary)-before_data[item])/(before_data[item]*self.N_0)

            self.measure_data[item][7].append(abs((boundary)-before_data[item])/(before_data[item]*self.N_0))
            self.measure_data[item][8].append((before_data[item]-(boundary))/(before_data[item]*self.N_0))


          #influx
          elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
            self.matrix[time][self.N_0][check_data[item]-1]+=1
            self.enter_SCI[time]+=abs((boundary)-check_data[item])/(check_data[item]*self.N_0)
            self.move_rank[1][check_data[item]-1]+=1
            self.fluc[time]+=abs(check_data[item]-(boundary))/(check_data[item]*self.N_0)
            self.fluc_1[time]+=((boundary)-check_data[item])/(check_data[item]*self.N_0)
            self.up[time]+=abs((boundary)-check_data[item])/(check_data[item]*self.N_0)

            self.measure_data[item][7].append(abs(check_data[item]-(boundary))/((check_data[item]*self.N_0)))
            self.measure_data[item][8].append(((boundary)-check_data[item])/(check_data[item]*self.N_0))

            if self.measure_data[item][9]>0:
              self.measure_data[item][10].append(check_data[item]-1)
              self.re_influx[time]+=((boundary)-check_data[item])/(check_data[item]*self.N_0)
            else:
              self.new_influx[time]+=((boundary)-check_data[item])/(check_data[item]*self.N_0)
              self.new_o_list[time]+=1
            self.measure_data[item][9]+=1


          else:
            self.measure_data[item][7].append(np.nan)
            self.measure_data[item][8].append(np.nan)

        if time==(self.T-1):
          r_data=np.array(self.measure_data[item][6])
          r_data=r_data[~np.isnan(r_data)]
          self.measure_data[item][4]=r_data.mean()
          self.measure_data[item][5]=r_data.std()

          i_data=np.array(self.measure_data[item][8])
          i_data=i_data[~np.isnan(i_data)]
          self.mean_i.append(i_data.mean())
          self.std_i.append(i_data.std())
          self.measure_data[item].append(i_data.mean())
          self.measure_data[item].append(i_data.std())

      before_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))
      if self.fluc[time]*self.N_0>=self.N_0:
        self.over.append(time)

    for i in range(self.N_0+1):
      self.matrix[:][:,i]/=np.sum(np.sum(self.matrix,axis=0)[i])


    self.Delta_r=[]
    self.Delta_i=[]

    for i in range(self.N_0+1):
      test=np.array([(i+1)-j for j in range(1,self.N_0+2)])
      test_i=np.array([(i+1-j)/min((i+1),j) for j in range(1,self.N_0+2)] )
      self.Delta_r.append(np.dot(np.sum(self.matrix,axis=0)[i],test.T))
      self.Delta_i.append(np.dot(np.sum(self.matrix,axis=0)[i],test_i.T))

    self.Delta_r[-1]=self.Delta_r[-1]/sum(np.sum(self.matrix,axis=0)[-1])
    self.Delta_i[-1]=self.Delta_i[-1]/sum(np.sum(self.matrix,axis=0)[-1])
    self.beta()




  def change_N_0(self,N_0=False):       # Used to analyze variations depending on N_0
    if N_0==False:
      N_0=self.N_0
    data=self.data.loc[self.data.Rank<=N_0]

    o_list=np.zeros(self.T)
    F_list=np.zeros(self.T)
    before_data=set(data[:N_0].Name)
    for i in range(self.T):
      o_list[i]=len(data[:N_0*(i+1)].Name.unique())/N_0
      check_data=set(data[N_0*(i):N_0*(i+1)].Name)
      F_list[i]=len(check_data-before_data)/N_0
      before_data=check_data.copy()
    mean_F=sum(F_list)/(self.T-1)
    dot_o=(o_list[-1]-o_list[0])/(self.T-1)
    return F_list,mean_F,o_list,dot_o


  def max_min(self,N_0=False,boundary=False):  # Used to examine variations depending on N_0 and the boundary parameter
    print('start')
    if N_0==False:
        N_0=self.N_0

    if boundary==False:
        boundary=self.N_0+1
    print(N_0, boundary)
    tot_SCI=np.zeros(self.T)
    abs_SCI=np.zeros(self.T)
    enter_SCI=np.zeros(self.T)
    out_SCI=np.zeros(self.T)
    inside_SCI=np.zeros(self.T)

    data_name=set(list(self.data.Name.unique()))

    before_data=dict(zip(self.data[:N_0].Name,self.data[:N_0].Rank)) # initial data
    for time in range(1,self.T):

      check_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time)+N_0].Name,self.data[self.N_0*(time):self.N_0*(time)+N_0].Rank)) # ranging ex) 1~250

      for item in data_name:
        if (item in list(check_data.keys())) and (item in list(before_data.keys())):
          tot_SCI[time]+=(before_data[item]-check_data[item])/(min(check_data[item],before_data[item]))
          abs_SCI[time]+=abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item]))
          inside_SCI[time]+=abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item]))

        elif not(item in list(check_data.keys())) and (item in list(before_data.keys())):
          tot_SCI[time]+=(before_data[item]-(boundary))/((before_data[item]))
          abs_SCI[time]+=abs(before_data[item]-(boundary))/((before_data[item]))
          out_SCI[time]+=abs(before_data[item]-(boundary))/((before_data[item]))

        elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
          tot_SCI[time]+=((boundary)-check_data[item])/((check_data[item]))
          abs_SCI[time]+=abs((boundary)-check_data[item])/((check_data[item]))
          enter_SCI[time]+=abs((boundary)-check_data[item])/((check_data[item]))
        try:
          if check_data[item] > N_0 :
            print(check_data[item])
        except:
          pass
        try:
          if before_data[item] > N_0 :
            print(before_data[item])
        except:
          pass

      before_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time)+N_0].Name,self.data[self.N_0*(time):self.N_0*(time)+N_0].Rank))

    return tot_SCI,abs_SCI,enter_SCI,out_SCI,inside_SCI


  def corr(self): ## Simply prints out various coefficients

    # print('out:',stats.pearsonr(self.out_SCI,self.fluc))
    # print('In:',stats.pearsonr(self.In_SCI,self.fluc))
    # print('enter:',stats.pearsonr(self.enter_SCI,self.fluc))
    outflux=stats.pearsonr(self.out_SCI,self.fluc)[0]
    inside=stats.pearsonr(self.In_SCI,self.fluc)[0]
    influx=stats.pearsonr(self.enter_SCI,self.fluc)[0]
    self.corrDF=pd.DataFrame({"outflux":outflux,"inside":inside,"influx":influx},index=[""])

  def delta_r(self):  # Measures rank fluctuations without applying any weights
    self.delta=np.zeros(self.T)
    self.enter_delta=np.zeros(self.T)
    self.out_delta=np.zeros(self.T)
    self.intra_delta=np.zeros(self.T)
    before_data=dict(zip(self.data[:self.N_0].Name,self.data[:self.N_0].Rank))
    data_name=set(list(self.data.Name.unique()))
    boundary=self.N_0+1
    for time in range(self.T):
      check_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))

      for item in data_name:

        if time!=0:
          if (item in list(check_data.keys())) and (item in list(before_data.keys())):

            self.delta[time]+=abs(check_data[item]-before_data[item])
            self.intra_delta[time]+=abs(check_data[item]-before_data[item])

          elif not(item in list(check_data.keys())) and (item in list(before_data.keys())):
            self.delta[time]+=abs((boundary)-before_data[item])
            self.out_delta[time]+=abs((boundary)-before_data[item])

          elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
            self.delta[time]+=abs(check_data[item]-(boundary))
            self.enter_delta[time]+=abs(check_data[item]-(boundary))

      before_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))
  
  
  # Tracks score fluctuations instead of rank changes 
  # (applicable only to Spotify data, since OTT data lacks score information)
  def delta_score(self):   
    self.delta_s=np.zeros(self.T)
    before_data=dict(zip(self.data[:self.N_0].Name,self.data[:self.N_0].Score))
    data_name=set(list(self.data.Name.unique()))
    boundary=self.N_0+1
    for time in range(self.T):
      check_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Score))

      for item in data_name:

        if time!=0:
          if (item in list(check_data.keys())) and (item in list(before_data.keys())):

            self.delta_s[time]+=abs(check_data[item]-before_data[item])

          elif not(item in list(check_data.keys())) and (item in list(before_data.keys())):
            self.delta_s[time]+=abs((boundary)-before_data[item])

          elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
            self.delta_s[time]+=abs(check_data[item]-(boundary))

      before_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Score))

  def R_S_SCI(self): # 스코어 계산
    self.ds=np.zeros(self.T)
    self.dr=np.zeros(self.T)
    self.dSCI=np.zeros(self.T)
    self.s_list=[]
    self.r_list=[]
    self.r_array=[]
    self.s_array=[]
    self.start_r=[]
    self.end_r=[]
    self.rb=[]
    self.rf=[]
    self.fx=[]
    N_0=self.N_0

    before_rank=dict(zip(self.data[:self.N_0].Name,self.data[:self.N_0].Rank))
    before_score=dict(zip(self.data[:self.N_0].Name,self.data[:self.N_0].Score))
    data_name=set(list(self.data.Name.unique()))
    boundary=self.N_0+1
    for time in range(self.T):
      check_score=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Score))
      check_rank=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))

      for item in data_name:

        if time!=0:
          if item in list(before_rank.keys()):
            self.r_array.append(before_rank[item])
            self.s_array.append(before_score[item])
          if (item in list(check_rank.keys())) and (item in list(before_rank.keys())):

            if abs(check_rank[item]-before_rank[item])!=0:
              self.ds[time]+=(abs(check_score[item]-before_score[item]))
              self.dr[time]+=(abs(check_rank[item]-before_rank[item]))
              self.s_list.append(abs(check_score[item]-before_score[item]))
              self.r_list.append(abs(check_rank[item]-before_rank[item]))
              self.dSCI[time]+=(abs(check_rank[item]-before_rank[item])/(min(check_rank[item],before_rank[item])*N_0))
              self.rb.append(before_rank[item])
              self.rf.append(check_rank[item])
              self.fx.append(1/(min(check_rank[item],before_rank[item])))

              self.start_r.append(before_rank[item])
              self.end_r.append(check_rank[item])


          elif not(item in list(check_rank.keys())) and (item in list(before_rank.keys())):
            self.ds[time]+=(abs(before_score[item]))
            self.dr[time]+=(abs((boundary)-before_rank[item]))
            self.s_list.append(abs(before_score[item]))
            self.r_list.append(abs((boundary)-before_rank[item]))
            self.dSCI[time]+=(abs((boundary)-before_rank[item])/((before_rank[item])*N_0))
            self.rb.append(before_rank[item])
            self.rf.append(boundary)
            self.fx.append(1/(min(boundary,before_rank[item])))

            self.start_r.append(before_rank[item])
            self.end_r.append(boundary)

          elif (item in list(check_rank.keys())) and not(item in list(before_rank.keys())):
            self.ds[time]+=(abs(check_score[item]))
            self.dr[time]+=(abs(check_rank[item]-(boundary)))

            self.s_list.append(abs(check_score[item]))
            self.r_list.append(abs(check_rank[item]-(boundary)))

            self.dSCI[time]+=(abs(check_rank[item]-(boundary))/((check_rank[item])*N_0))
            self.rb.append(boundary)
            self.rf.append(check_rank[item])
            self.fx.append(1/(min(check_rank[item],boundary)))

            self.start_r.append(boundary)
            self.end_r.append(check_rank[item])

      before_score=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Score))
      before_rank=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))
    
  def beta(self):
    def cal_beta(r_t0,r_t1,beta):
      if beta!=0:
        A=((r_t0**beta+r_t1**beta)/2)**(1/beta)
      else:
        A=(r_t0*r_t1)**(1/2)
      return abs(r_t0-r_t1)*(1/A)

    self.beta_0_in=np.zeros(self.T)
    self.beta_0_out=np.zeros(self.T)
    self.beta_0_inter=np.zeros(self.T)

    self.beta_1_in=np.zeros(self.T)
    self.beta_1_out=np.zeros(self.T)
    self.beta_1_inter=np.zeros(self.T)

    self.beta_m1_in=np.zeros(self.T)
    self.beta_m1_out=np.zeros(self.T)
    self.beta_m1_inter=np.zeros(self.T)

    boundary=self.N_0+1


    data_name=set(list(self.data.Name.unique()))

    # t 시간 데이터
    before_data=dict(zip(self.data[:self.N_0].Name,self.data[:self.N_0].Rank))

    for time in range(1,self.T):
      check_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))
      for item in data_name:
    # try:
        # t+1 시간 데이터
        #본격 SCI분석
        #inside
        
        if (item in list(check_data.keys())) and (item in list(before_data.keys())):
          self.beta_0_inter[time]+=cal_beta(before_data[item],check_data[item],0)
          self.beta_1_inter[time]+=cal_beta(before_data[item],check_data[item],1)
          self.beta_m1_inter[time]+=cal_beta(before_data[item],check_data[item],-1)

        #outflux
        elif not(item in list(check_data.keys())) and (item in list(before_data.keys())):

          self.beta_0_out[time]+=cal_beta(before_data[item],boundary,0)
          self.beta_1_out[time]+=cal_beta(before_data[item],boundary,1)
          self.beta_m1_out[time]+=cal_beta(before_data[item],boundary,-1)

        #influx
        elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
          self.beta_0_in[time]+=cal_beta(boundary,check_data[item],0)
          self.beta_1_in[time]+=cal_beta(boundary,check_data[item],1)
          self.beta_m1_in[time]+=cal_beta(boundary,check_data[item],-1)

      before_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))
    

  def item_high_SCI(self): #SCI 구조 변화를 무엇이 많이 주는지 확인하기 위해서 사용
    abs_SCI=np.zeros(self.T)
    boundary=self.N_0+1
    data_name=set(list(self.data.Name.unique()))
    self.prob=np.ones(self.T)
    self.prob_1=np.zeros(self.T)
    self.over_list=[]
    self.over_time=[]
    self.re_enter=dict(zip(np.arange(100),np.zeros(101)))
    for time in range(1,self.T):
      before_data=dict(zip(self.data[self.N_0*(time-1):self.N_0*(time)].Name,self.data[self.N_0*(time-1):self.N_0*(time)].Rank))
      check_data=dict(zip(self.data[self.N_0*(time):self.N_0*(time+1)].Name,self.data[self.N_0*(time):self.N_0*(time+1)].Rank))
      # if time ==0:
      #   print(before_data.keys())
      #   print(check_data.keys())
      for item in data_name:
        # print(np.sum(self.matrix,axis=0).shape())

        if time in self.over:
          if (item in list(check_data.keys())) and (item in list(before_data.keys())) :
            self.prob[time]*=np.sum(self.matrix,axis=0)[before_data[item]-1][check_data[item]-1]
            self.prob_1[time]-=np.log(np.sum(self.matrix,axis=0)[before_data[item]-1][check_data[item]-1])
            abs_SCI[time]+=abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item])*self.N_0)
            if abs(before_data[item]-check_data[item])/(min(check_data[item],before_data[item]))>= self.N_0*(1/3) and not (item in self.over_list):
              self.over_list.append(item)
              self.over_time.append(time)
            if np.sum(self.matrix,axis=0)[before_data[item]-1][check_data[item]-1]==0:
              print(time,before_data[item]-1,check_data[item]-1)
              print(item)
              print(1)

          elif not(item in list(check_data.keys())) and (item in list(before_data.keys())) :
            self.prob[time]*=np.sum(self.matrix,axis=0)[before_data[item]-1][boundary-1]
            self.prob_1[time]-=np.log(np.sum(self.matrix,axis=0)[before_data[item]-1][boundary-1])
            abs_SCI[time]+=abs(before_data[item]-(boundary))/((before_data[item])*self.N_0)
            if abs(before_data[item]-(boundary))/((before_data[item]))>= self.N_0**(1/3) and not (item in self.over_list):
              self.over_list.append(item)
              self.over_time.append(time)
            if np.sum(self.matrix,axis=0)[before_data[item]-1][boundary-1]==0:
              print(time,before_data[item]-1,boundary-1)
              print(item)
              print(2)

          elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
            self.prob[time]*=np.sum(self.matrix,axis=0)[boundary-1][check_data[item]-1]
            self.prob_1[time]-=np.log(np.sum(self.matrix,axis=0)[boundary-1][check_data[item]-1])
            abs_SCI[time]+=abs((boundary)-check_data[item])/((check_data[item])*self.N_0)
            if abs((boundary)-check_data[item])/((check_data[item]))>= self.N_0**(1/3) and not (item in self.over_list):
              self.over_list.append(item)
              self.over_time.append(time)
            if np.sum(self.matrix,axis=0)[boundary-1][check_data[item]-1]==0:
              print(time,boundary-1,check_data[item]-1)
              print(item)
              print(3)
        else:
          if (item in list(check_data.keys())) and (item in list(before_data.keys())) :
            self.prob[time]*=np.sum(self.matrix,axis=0)[before_data[item]-1][check_data[item]-1]
            self.prob_1[time]-=np.log(np.sum(self.matrix,axis=0)[before_data[item]-1][check_data[item]-1])
            if np.sum(self.matrix,axis=0)[before_data[item]-1][check_data[item]-1]==0:
              print(time,before_data[item]-1,check_data[item]-1)
              print(item)
              print(4)

          elif not(item in list(check_data.keys())) and (item in list(before_data.keys())) :
            self.prob[time]*=np.sum(self.matrix,axis=0)[before_data[item]-1][boundary-1]
            self.prob_1[time]-=np.log(np.sum(self.matrix,axis=0)[before_data[item]-1][boundary-1])
            if np.sum(self.matrix,axis=0)[before_data[item]-1][boundary-1]==0:
              print(time,before_data[item]-1,boundary-1)
              print(item)
              print(5)

          elif (item in list(check_data.keys())) and not(item in list(before_data.keys())):
            self.prob[time]*=np.sum(self.matrix,axis=0)[boundary-1][check_data[item]-1]
            self.prob_1[time]-=np.log(np.sum(self.matrix,axis=0)[boundary-1][check_data[item]-1])
            if np.sum(self.matrix,axis=0)[boundary-1][check_data[item]-1]==0:
              print(time,boundary-1,check_data[item]-1)
              print(item)
              print(6)

    max_num=0
    for item in data_name:
      self.re_enter[self.measure_data[item][9]]+=1
      if max_num<=self.measure_data[item][9]:
        self.re_enter_max=item

def barplot(A,B,C,D,E,F):
  plt.figure(figsize=(15,10))
  i=1
  tit=['Netflix','Disney','Hubo','Prime','paramaount','Hulu']
  for Name in [A,B,C,D,E,F]:

    plt.subplot(3,2,i)
    p1 = plt.bar(np.arange(len(Name[0])),Name[0], color='green')
    p2 = plt.bar( np.arange(len(Name[0])), Name[1], color='dodgerblue', bottom=Name[0])
    p3 = plt.bar( np.arange(len(Name[0])), Name[2] , color='orange', bottom=Name[0]+Name[1])
    plt.title(tit[i-1])
    i+=1
  plt.show()





def cut(A,B,C):

  bins=np.arange(0,1.01,0.1)
  tot=len(A)
  cut_A = pd.cut(A, bins=bins)
  cut_B = pd.cut(B, bins=bins)
  cut_C = pd.cut(C, bins=bins)

  v_A=np.array(cut_A.value_counts().sort_index())/tot
  v_B=np.array(cut_B.value_counts().sort_index())/tot
  v_C=np.array(cut_C.value_counts().sort_index())/tot
  return v_A,v_B,v_C,np.delete(bins,0)

def cut1(A,B,n):
  cut_A=[0]
  cut_B=[0]
  num=(len(A)/n)
  for i in range(n):

    cut_A.append(sum(A[int(num*i):int(num*(i+1))]))
    cut_B.append(sum(B[int(num*i):int(num*(i+1))]))
  bins=np.arange(0,1.01,1/n)
  return cut_A,cut_B,bins

def cut2(A,B,n):
  cut_A=[0]
  cut_B=[0]
  num=(len(A)/n)
  for i in range(n):

    cut_A.append(cut_A[-1]+sum(A[int(num*i):int(num*(i+1))]))
    cut_B.append(cut_B[-1]+sum(B[int(num*i):int(num*(i+1))]))
  bins=np.arange(0,1.01,1/n)
  return cut_A,cut_B,bins

def concatenate_names_city(row):
    return f"{row['Track Name']}-{row['Artist']}"

def s(array,name):
  np.save("{:s}.npy".format(name),array)

def N_b_alpha(A,n_data):
  per=np.array([0,0.25,0.5,0.75,1])
  for N_b_per in per[1:]:
    for alpha in per:
      N_b=int(A.N_0*N_b_per)
      alpha1=N_b+int(A.N_0*alpha)+1
      tot_SCI,abs_SCI,enter_SCI,out_SCI,inside_SCI=A.max_min(N_b,alpha1)
       
      np.save(save_data+n_data+f'tot_{N_b}_{alpha1}',tot_SCI)
      np.save(save_data+n_data+f'abs_{N_b}_{alpha1}',abs_SCI)
      np.save(save_data+n_data+f'enter_{N_b}_{alpha1}',enter_SCI)
      np.save(save_data+n_data+f'out_{N_b}_{alpha1}',out_SCI)
      np.save(save_data+n_data+f'intra_{N_b}_{alpha1}',inside_SCI)



def all_save(p_data,n_data): 
  N_b_alpha(p_data,n_data)
  s(p_data.up*p_data.N_0,save_data+n_data+'_up')
  s(p_data.down*p_data.N_0,save_data+n_data+'_down')
  s(p_data.F_list,save_data+n_data+'_F_list')

  s(p_data.fluc*p_data.N_0,save_data+n_data+'_flux')
  s(p_data.In_SCI*p_data.N_0,save_data+n_data+'_Interflux')
  s(p_data.enter_SCI*p_data.N_0,save_data+n_data+'_Influx')
  s(p_data.out_SCI*p_data.N_0,save_data+n_data+'_outflux')

  f, P = scipy.signal.periodogram(p_data.fluc*p_data.N_0, p_data.T, nfft=2**12)
  s(f,save_data+n_data+'_f')
  s(P,save_data+n_data+'_p')

  A,B,bin=cut1(p_data.move_rank[0]/sum(p_data.move_rank[0]),p_data.move_rank[1]/sum(p_data.move_rank[1]),10)
  s(A,save_data+n_data+'_pdf_out')
  s(B,save_data+n_data+'_pdf_in')
  s(bin,save_data+n_data+'_pdf_bin')

  A,B,bin=cut2(p_data.move_rank[0]/sum(p_data.move_rank[0]),p_data.move_rank[1]/sum(p_data.move_rank[1]),10)
  s(A,save_data+n_data+'_cdf_out')
  s(B,save_data+n_data+'_cdf_in')
  s(bin,save_data+n_data+'_cdf_bin')
  s(p_data.new_o_list,save_data+n_data+'_newo')

  s(p_data.matrix,save_data+n_data+'_matrix')
  s(p_data.N_0,save_data+n_data+'_N')
  s(p_data.N_last,save_data+n_data+'_Nlast')
  s(p_data.prob,save_data+n_data+'_prob')
  s(p_data.prob_1,save_data+n_data+'_prob1')

  s(p_data.beta_0_in,save_data+n_data+'_beta_0_in')
  s(p_data.beta_0_out,save_data+n_data+'_beta_0_out')
  s(p_data.beta_0_inter,save_data+n_data+'_beta_0_inter')
  s(p_data.beta_1_in,save_data+n_data+'_beta_1_in')
  s(p_data.beta_1_out,save_data+n_data+'_beta_1_out')
  s(p_data.beta_1_inter,save_data+n_data+'_beta_1_inter')
  s(p_data.beta_m1_in,save_data+n_data+'_beta_m1_in')
  s(p_data.beta_m1_out,save_data+n_data+'_beta_m1_out')
  s(p_data.beta_m1_inter,save_data+n_data+'_beta_m1_inter')
  
  s(p_data.measure_data,save_data+n_data+'_measure_data')

  if not(n_data in ['netflix','Disney','Hbo','prime']):
    s(p_data.r_array,save_data+n_data+'_r_array')
    s(p_data.s_array,save_data+n_data+'_s_array')
    s(p_data.rb,save_data+n_data+'_rb')
    s(p_data.rf,save_data+n_data+'_rf')
    s(p_data.dSCI,save_data+n_data+'_dSCI')
    s(p_data.dr,save_data+n_data+'_dr')
    s(p_data.ds,save_data+n_data+'_ds')
    s(p_data.r_list,save_data+n_data+'_dr_list')
    s(p_data.s_list,save_data+n_data+'_ds_list')
    s(p_data.intra_delta,save_data+n_data+'_dr_intra')
    s(p_data.out_delta,save_data+n_data+'_dr_out')
    s(p_data.enter_delta,save_data+n_data+'_dr_enter')
  else:
    s(p_data.delta,save_data+n_data+'_dr')
    s(p_data.intra_delta,save_data+n_data+'_dr_intra')
    s(p_data.out_delta,save_data+n_data+'_dr_out')
    s(p_data.enter_delta,save_data+n_data+'_dr_enter')

load_data='./ranking/'
save_data='./data/'
tag=str(sys.argv[1])

# Lord data ======================================================================================================================================================================

if tag=='netflix':
  World_movie_df=pd.read_csv(load_data+'Netflix_World_Movie.csv',index_col=0)
  World_movie_df=World_movie_df.drop(columns=['2023-01-01'])
  World_movie_df.tail()
  full_netflix= World_movie_df.copy()

  #데이터 전처리
  copy_data_netflix=World_movie_df.dropna().copy()
  copy_data_netflix=copy_data_netflix.reset_index().rename(columns={'index':'Rank'})

  t=len(World_movie_df.columns)
  date_list=copy_data_netflix.columns[1:]
  new_data=pd.DataFrame({'Date':date_list[0],
                        'Name':copy_data_netflix[date_list[0]],
                        'Rank':copy_data_netflix['Rank']},)
  for i in range(1,t):
    new_data=pd.concat([new_data,pd.DataFrame({'Date':date_list[i],
                        'Name':copy_data_netflix[date_list[i]],
                        'Rank':copy_data_netflix['Rank']},)],axis=0)
  netflix_data1=new_data.rename(columns={'Date':'T'})
  netflix_data1.tail()

  netflix=Analysis(netflix_data1)

  netflix.measure()
  netflix.item_high_SCI()
  netflix.delta_r()
  netflix.beta()

  all_save(netflix,'netflix')
  print(tag,'done')

elif tag=='hbo':
#데이터 불러오기
  World_movie_df=pd.read_csv(load_data+'Hbo_World_Movie.csv',index_col=0)
  World_movie_df.tail()

  #데이터 전처리

  copy_data_netflix=World_movie_df.dropna().copy()
  copy_data_netflix=copy_data_netflix.reset_index().rename(columns={'index':'Rank'})

  t=len(World_movie_df.columns)
  date_list=copy_data_netflix.columns[1:]
  new_data=pd.DataFrame({'Date':date_list[0],
                        'Name':copy_data_netflix[date_list[0]],
                        'Rank':copy_data_netflix['Rank']},)
  for i in range(1,t):
    new_data=pd.concat([new_data,pd.DataFrame({'Date':date_list[i],
                        'Name':copy_data_netflix[date_list[i]],
                        'Rank':copy_data_netflix['Rank']},)],axis=0)
  Hbo_data1=new_data.rename(columns={'Date':'T'})
  Hbo_data1.tail()

  Hbo=Analysis(Hbo_data1)

  Hbo.measure()
  Hbo.item_high_SCI()
  Hbo.delta_r()
  Hbo.beta()
  all_save(Hbo,'Hbo')
  print(tag,'done')

#데이터 불러오기
elif tag=='disney':
  World_movie_df=pd.read_csv(load_data+'Disney_World_Movie.csv',index_col=0)
  World_movie_df.tail()

  #데이터 전처리
  copy_data_netflix=World_movie_df.dropna().copy()
  copy_data_netflix=copy_data_netflix.reset_index().rename(columns={'index':'Rank'})

  t=len(World_movie_df.columns)
  date_list=copy_data_netflix.columns[1:]
  new_data=pd.DataFrame({'Date':date_list[0],
                        'Name':copy_data_netflix[date_list[0]],
                        'Rank':copy_data_netflix['Rank']},)

  for i in range(1,t):
    new_data=pd.concat([new_data,pd.DataFrame({'Date':date_list[i],
                        'Name':copy_data_netflix[date_list[i]],
                        'Rank':copy_data_netflix['Rank']},)],axis=0)
  Disney_data1=new_data.rename(columns={'Date':'T'})

  Disney_data1.tail()

  Disney=Analysis(Disney_data1)

  Disney.measure()
  Disney.item_high_SCI()
  Disney.delta_r()
  Disney.beta()
  all_save(Disney,'Disney')
  print(tag,'done')

elif tag=='prime':
#데이터 불러오기
  World_movie_df=pd.read_csv(load_data+'prime_World_Movie.csv',index_col=0)
  World_movie_df.tail()

  #데이터 전처리

  copy_data_netflix=World_movie_df.dropna().copy()
  copy_data_netflix=copy_data_netflix.reset_index().rename(columns={'index':'Rank'})

  t=len(World_movie_df.columns)
  date_list=copy_data_netflix.columns[1:]
  new_data=pd.DataFrame({'Date':date_list[0],
                        'Name':copy_data_netflix[date_list[0]],
                        'Rank':copy_data_netflix['Rank']},)
  for i in range(1,t):
    new_data=pd.concat([new_data,pd.DataFrame({'Date':date_list[i],
                        'Name':copy_data_netflix[date_list[i]],
                        'Rank':copy_data_netflix['Rank']},)],axis=0)
  prime_data1=new_data.rename(columns={'Date':'T'})
  prime_data1.tail()

  prime=Analysis(prime_data1)

  prime.measure()
  prime.item_high_SCI()
  prime.delta_r()
  prime.beta()
  all_save(prime,'prime')
  print(tag,'done')


def spotify_chekc(data,N_tot,tag):
    first= len(data[data=='2017-01-01']['Date'].dropna())


    # 새로운 'Name_City' 열 추가
    data['Track Name_artist'] = data.apply(concatenate_names_city, axis=1)

    new_data=pd.DataFrame({'Date':'2017-01-01',
                          'Name':data[data['Date']=='2017-01-01']['Track Name_artist'][:N_tot],
                          'Artist': data[data['Date']=='2017-01-01']['Artist'][:N_tot],
                          'Rank':data.index[:N_tot],
                          'Score':data[data['Date']=='2017-01-01']['Streams'][:N_tot]})
    empty_day=[]
    for i in range(len(date_arange)-1):
      date=date_arange[i+1]
      
      if date not in existing_dates: 
        print('not in')
        date_1=date_last
        empty_day.append(date)
        
        new_data=pd.concat([new_data,pd.DataFrame({'Date':date,
                            'Name':data[data['Date']==date_1]['Track Name_artist'][:N_tot],
                            'Artist': data[data['Date']==date_1]['Artist'][:N_tot],
                            'Rank':data.index[:N_tot],
                            'Score':data[data['Date']==date_1]['Streams'][:N_tot]})],axis=0)
      
      else:
        date_last=date 
        # print(date)
        new_data=pd.concat([new_data,pd.DataFrame({'Date':date,
                            'Name':data[data['Date']==date]['Track Name_artist'][:N_tot],
                            'Artist': data[data['Date']==date]['Artist'][:N_tot],
                            'Rank':data.index[:N_tot],
                            'Score':data[data['Date']==date]['Streams'][:N_tot]})],axis=0)

    print('empty day:',empty_day)

    spotify_data1=new_data.rename(columns={'Date':'T'})

    spotify_data1.to_csv(f'/home/woo/ott/check_data/data_{tag}.csv')
    spotify=Analysis(spotify_data1)

    spotify.measure()
    spotify.delta_r()
    spotify.delta_score()
    spotify.item_high_SCI()
    spotify.R_S_SCI()
    spotify.beta()

    all_save(spotify,f'{tag}_spotify')
    print(tag,'done')

# @title spotify
#데이터 불러오기
spotify_df=pd.read_csv(load_data+'spotify_rankig.csv',index_col=0)
spotify_df.tail()
full_spotify= spotify_df.copy()

full_spotify.head()


#스웨덴 일본
spotify_se=full_spotify[full_spotify['Region']=='se']
spotify_jp=full_spotify[full_spotify['Region']=='jp']
spotify_us=full_spotify[full_spotify['Region']=='us']
spotify_au=full_spotify[full_spotify['Region']=='au']
spotify_global=full_spotify[full_spotify['Region']=='global']
spotify_tw=full_spotify[full_spotify['Region']=='tw']
spotify_gb=full_spotify[full_spotify['Region']=='gb']

start_date=min(full_spotify['Date'].unique())
end_date=max(full_spotify['Date'].unique())
print(start_date,end_date)

date_arange = pd.date_range(start=start_date, end=end_date)
existing_dates = full_spotify['Date'].unique()
date_arange = [d.strftime('%Y-%m-%d') for d in date_arange]
if tag=='sw':
  spotify_chekc(spotify_se,200,tag)

elif tag=='jp':
  spotify_chekc(spotify_jp,108,tag)

elif tag=='us':
  spotify_chekc(spotify_us,108,tag)

elif tag=='au':
  spotify_chekc(spotify_au,108,tag)

elif tag=='tw':
  spotify_chekc(spotify_tw,200,tag)

elif tag=='gb':
  spotify_chekc(spotify_gb,200,tag)

