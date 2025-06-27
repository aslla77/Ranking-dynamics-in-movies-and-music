
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import pycountry
from datetime import datetime
fsize=45



def boxp1(C,N,Max_r=True,include=True,rank_life=True,legen=False):
    x_values = []  # 첫 번째 값 (binning 기준)
    y_values = []  # 세 번째 값 (boxplot 대상)
    max_ES_L=0
    max_life=C['max_life']
    count=0
    if include:
        if rank_life:
            print('include, rank life')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf):
                    y_values.append(C[i][0])
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
        else:
            print('include, End-Start')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf):
                    s=datetime.strptime(C[i][5], '%Y-%m-%d')
                    e=datetime.strptime(C[i][6], '%Y-%m-%d')
                    day_gap=e-s
                    y_values.append(day_gap.days)
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
                    if max_ES_L< day_gap.days:
                        max_ES_L=day_gap.days
    else:
        if rank_life:
            print('exclude, rank life')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                    y_values.append(C[i][0])
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
        else:
            print('exclude, End-Start')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                    s=datetime.strptime(C[i][5], '%Y-%m-%d')
                    e=datetime.strptime(C[i][6], '%Y-%m-%d')
                    day_gap=e-s
                    y_values.append(day_gap.days)
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
                    if max_ES_L< day_gap.days:
                        max_ES_L=day_gap.days


    # print(count)
    # Binning 설정 (예: 3개 구간으로 나누기)
    num_bins = N
    bins = np.linspace(0, max(x_values), num_bins + 1)  # 구간 설정
    test = np.linspace(0, 100, num_bins + 1)  # 구간 설정
    bin_labels = [f"{int(test[i])+1}\u2013{int(test[i+1])}" for i in range(len(bins)-1)]

    # 각 bin에 대한 2번 데이터 저장
    binned_data = {label: [] for label in bin_labels}

    # 데이터를 해당 bin에 할당
    for x, y in zip(x_values, y_values):
        for i in range(len(bins) - 1):
            if bins[i] <= x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                binned_data[bin_labels[i]].append(y)
                break

    box=plt.boxplot(binned_data.values(), labels=np.arange(1,11), boxprops=dict(linewidth=2.5), # notch=True,
            showmeans=True, meanline=True, 
            whiskerprops=dict(color='gray', linestyle='--', linewidth=2.5),
            capprops=dict(color='black', linewidth=2.5),
            medianprops=dict(color='blue', linestyle='--', linewidth=5),
            meanprops=dict(color= 'red', linestyle='-', linewidth= 5))


    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    if legen!=False:
        plt.legend([box['medians'][0], box['means'][0]],['Median', 'Mean'], loc='best', fontsize=fsize, frameon=False)
    



def boxp2(C,N,Max_r=True,include=True,rank_life=True,legen=False):
    x_values = []  # 첫 번째 값 (binning 기준)
    y_values = []  # 세 번째 값 (boxplot 대상)
    max_ES_L=0
    max_life=C['max_life']
    count=0
    if include:
        if rank_life:
            print('include, rank life')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf):
                    y_values.append(C[i][0])
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
        else:
            print('include, End-Start')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf):
                    s=datetime.strptime(C[i][5], '%Y-%m-%d')
                    e=datetime.strptime(C[i][6], '%Y-%m-%d')
                    day_gap=e-s
                    y_values.append(day_gap.days)
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
                    if max_ES_L< day_gap.days:
                        max_ES_L=day_gap.days
    else:
        if rank_life:
            print('exclude, rank life')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                    y_values.append(C[i][0])
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
        else:
            print('exclude, End-Start')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                    s=datetime.strptime(C[i][5], '%Y-%m-%d')
                    e=datetime.strptime(C[i][6], '%Y-%m-%d')
                    day_gap=e-s
                    y_values.append(day_gap.days)
                    count+=1
                    if Max_r:
                        x_values.append(C[i][2])
                    else:    
                        x_values.append(C[i][1])
                    if max_ES_L< day_gap.days:
                        max_ES_L=day_gap.days


    # print(count)
    # Binning 설정 (예: 3개 구간으로 나누기)
    num_bins = N
    bins = np.linspace(0, max(x_values), num_bins + 1)  # 구간 설정
    test = np.linspace(0, 100, num_bins + 1)  # 구간 설정
    bin_labels = [f"{int(test[i])} ~ {int(test[i+1])}" for i in range(len(bins)-1)]

    # 각 bin에 대한 2번 데이터 저장
    binned_data = {label: [] for label in bin_labels}

    # 데이터를 해당 bin에 할당
    for x, y in zip(x_values, y_values):
        for i in range(len(bins) - 1):
            if bins[i] <= x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                binned_data[bin_labels[i]].append(y)
                break
    
    
    for i in range(len(binned_data)):
        data=binned_data[bin_labels[i]]
        q1, median, q3 = np.percentile(data, [25, 50, 75])  # 사분위수 계산
        mean = np.mean(data)  # 평균 계산
        min_val, max_val = np.min(data), np.max(data)  # 최솟값, 최댓값

        # 수평선 (위/아래 끝에 추가)
        plt.plot([(i+1) - 0.2, (i+1) + 0.2], [q1, q1], color='black', linewidth=2)  # 하단 수평선
        plt.plot([(i+1) - 0.2, (i+1) + 0.2], [q3, q3], color='black', linewidth=2)  # 상단 수평선

        # 중앙 사분위 영역
        plt.plot([(i+1), (i+1)], [q1, q3], color='black', linewidth=2)  # 두꺼운 중앙 영역

        # 평균과 중앙값 표시
        plt.scatter((i+1), mean, color='red', marker='x', s=500, linewidths=3, edgecolors='black', label="Mean" if i == 1 else "")  # 평균(X)
        plt.scatter((i+1), median, color='blue', marker='o', s=500, linewidths=3, edgecolors='black', label="Median" if i == 1 else "")  # 중앙값(O)

    plt.ylabel("life_time",fontsize=fsize)
    plt.xticks(ticks=range(1,11), labels=np.arange(1,11), fontsize=30)# ,rotation=45)
    plt.tick_params(width=3,  length=5, pad=6, labelsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.grid(True)
    if legen!=False:
        plt.legend( loc='best', fontsize=fsize, frameon=False)
    

def binning2(C ,N,log=False,include=True,rank_life=True,x_label=False,y_label=False):
    S=datetime.strptime(C['S'], '%Y-%m-%d')
    E=datetime.strptime(C['E'], '%Y-%m-%d')
    test=[]
    max_r=C['max_r']
    count=0
    if include: 
        if rank_life:
            life_dis=np.zeros((C['max_life']+1))
            print('include, rank life')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf):
                    life_dis[C[i][0]]+=1
                    count+=1
                    test.append(C[i][0])
        else:
            life_dis=np.zeros((C['T']+1))
            print('include, End-Start')
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf):
                    s=datetime.strptime(C[i][5], '%Y-%m-%d')
                    e=datetime.strptime(C[i][6], '%Y-%m-%d')
                    day_gap=e-s
                    life_dis[day_gap.days]+=1
                    test.append(day_gap.days)
                    count+=1
    else:
        if rank_life:
            print('exclude, rank life')
            life_dis=np.zeros((C['max_life']+1))
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                    life_dis[C[i][0]]+=1
                    test.append(C[i][0])
                    count+=1
        else:
            print('exclude, End-Start')
            life_dis=np.zeros((C['T']+1))
            for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
                x=C[i][0]
                y=C[i][1]
                z=C[i][2]
                if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                    s=datetime.strptime(C[i][5], '%Y-%m-%d')
                    e=datetime.strptime(C[i][6], '%Y-%m-%d')
                    day_gap=e-s
                    life_dis[day_gap.days]+=1
                    test.append(day_gap.days)
                    count+=1
    test=np.array(test)
    test+=1
    life_dis+=1
    life_dis=life_dis/count
    # print(count)
    # Binning 설정 (예: 3개 구간으로 나누기)

    num_bins = N
    if not(log):
        if rank_life:
            bins = np.linspace(0,  C['max_life']+1, num_bins + 1)  # 구간 설정
            x_values=np.arange(C['max_life']+1)
        else:
            bins = np.linspace(0,  C['T']+1, num_bins + 1)  # 구간 설정
            x_values=np.arange(C['T']+1)
        
        print('bin:',bins)
        bin_labels = [f"{int(bins[i])} ~ {int(bins[i+1])}" for i in range(len(bins)-1)]

        # 각 bin에 대한 2번 데이터 저장
        binned_data = {label: [] for label in bin_labels}

        # 데이터를 해당 bin에 할당
        for x, y in zip(x_values, life_dis):
            for i in range(len(bins) - 1):
                if bins[i] < x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                    binned_data[bin_labels[i]].append(y)
                    break

        # X축 (bin 범위), Y축 (각 bin의 평균값)
        bins = list(binned_data.keys())  # 범위 (문자열)
        means = [np.mean(np.array(values)) for values in binned_data.values()]  # 각 bin의 평균값 계산


        plt.plot(bins, means, marker='o', linestyle='-', color='b', label='Mean')
    #------------------------
    else:
        data=test.copy()
        min_k=np.min(data)
        max_k=np.max(data)

        dist=np.bincount(data)[1:]/len(data)
        # print('distsum:',np.sum(dist))
        axis=np.arange(min_k,max_k+1)
        # print('min:',min_k,'max_k:',max_k)
        edge_of_bins=np.logspace(np.log(min_k),np.log(max_k),N,base=np.e,endpoint=True)
        # print(edge_of_bins)
        print(edge_of_bins)

        max_index=np.max((np.where(edge_of_bins<10)[0]))
        max_digit=edge_of_bins[max_index]
        single_digit=np.arange(0.5,9.6,1)
        edge_of_bins=np.delete(edge_of_bins,np.where(edge_of_bins<10)[0])
        print('2',edge_of_bins)

        edge_of_bins=np.hstack((single_digit, edge_of_bins))
        print('3',edge_of_bins)
        
        hist,bins=np.histogram(data,edge_of_bins,density=True)
        # print('bins:',bins)
        # print(bins)
        hist=np.zeros(len(hist))
        count=0
        for x in data:
            for i in range(len(edge_of_bins)-1):
                if edge_of_bins[i]<=x < edge_of_bins[i+1]:
                    hist[i]+=1/(edge_of_bins[i+1]- edge_of_bins[i])
                    count+=1
                    break 

        hist=hist/count
        axis2=0.5*(edge_of_bins[1:]+edge_of_bins[:-1])
        # print(count)

        plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor='red', # 마커 내부 색깔
    markeredgecolor='red',
    lw=3,
    ms=10)
        plt.scatter(axis, dist, marker='o', linestyle='-', color='gray',alpha=0.2)
        total=0
    # ------------------------
    plt.yscale('log')
    plt.xscale('log')
    if y_label:
        plt.ylabel("PDF",fontsize=fsize)
    if x_label:
        plt.xlabel("Lifetime",fontsize=fsize)
    plt.xticks(fontsize=fsize)  # x축 틱 크기 조절
    plt.yticks(fontsize=fsize)




def Max_binning(C ,N,log=False,include=True,x_label=False,y_label=False):
    print('Max_binning')
    S=datetime.strptime(C['S'], '%Y-%m-%d')
    E=datetime.strptime(C['E'], '%Y-%m-%d')
    test=[]
    max_r=C['max_r']
    count=0
    if include: 
        life_dis=np.zeros((C['max_r']+1))
        print('include')
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf):
                life_dis[C[i][2]]+=1
                count+=1
                test.append(C[i][2])
                count+=1
    else:
        print('exclude')
        life_dis=np.zeros((C['max_r']+1))
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                life_dis[C[i][2]]+=1
                test.append(C[i][2])
                count+=1
    life_dis=life_dis/count
    # Binning 설정 (예: 3개 구간으로 나누기)
    print(1,C['max_r'],len(life_dis))

    num_bins = N
    if not(log):
        bins = np.linspace(0,  C['max_r']+2, num_bins + 1)  # 구간 설정
        x_values=np.arange(C['max_r']+2)
        
        bin_labels = [f"{int(bins[i])} ~ {int(bins[i+1])}" for i in range(len(bins)-1)]

        # 각 bin에 대한 2번 데이터 저장
        binned_data = {label: [] for label in bin_labels}

        # 데이터를 해당 bin에 할당
        for x, y in zip(x_values, life_dis):
            for i in range(len(bins) - 1):
                if bins[i] < x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                    binned_data[bin_labels[i]].append(y)
                    break

        # X축 (bin 범위), Y축 (각 bin의 평균값)
        bins = list(binned_data.keys())  # 범위 (문자열)
        means = [np.mean(np.array(values)) for values in binned_data.values()]  # 각 bin의 평균값 계산


        plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor='red', # 마커 내부 색깔
    markeredgecolor='red',
    lw=3,
    ms=10)
        plt.scatter(axis, dist, marker='o', linestyle='-', color='gray',alpha=0.2)
    #------------------------
    else:
        try:
            data=test.copy()
            min_k=np.min(data)
            max_k=np.max(data)

            dist=np.bincount(data, minlength=C['max_r']+1)/len(data)
            axis=np.arange(min_k-0.5,max_k+1.5,C['max_r']/N) 
            hist,bins=np.histogram(data,axis,density=True)
            hist=np.zeros(len(hist))
            count=0
            for x in data:
                for i in range(len(axis)-1):
                    if axis[i]<x <= axis[i+1]:
                        hist[i]+=1/(axis[i+1]- axis[i])
                        count+=1
                        break 

            hist=hist/count
            axis2=0.5*(axis[1:]+axis[:-1])
            # print(count)


            plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
        color='black',          # 선 색깔
        marker='o',            # 마커 모양
        markerfacecolor='red', # 마커 내부 색깔
        markeredgecolor='red',
        lw=3,
        ms=10)
            plt.scatter(np.arange(C['max_r']+1), dist, marker='o', linestyle='-', color='gray',alpha=0.2)
        except:
            print(2,len(np.arange(C['max_r']+1)),len(dist))
            print(len(axis2),len(hist))
    # ------------------------
    check=0
    for i in range(len(axis)-1):
        check+=hist[i]*(axis[i+1]-axis[i])
    print(check)
    if not(log):
        plt.yscale('log')
        plt.xscale('log')
    if y_label:
        plt.ylabel("PDF",fontsize=fsize)
    if x_label:
        plt.xlabel("Highest Rank",fontsize=fsize)
    plt.xticks(fontsize=fsize)  # x축 틱 크기 조절
    plt.yticks(fontsize=fsize)




def first_binning(C ,N,log=False,include=True,x_label=False,y_label=False,c='red'):
    print('first')
    S=datetime.strptime(C['S'], '%Y-%m-%d')
    E=datetime.strptime(C['E'], '%Y-%m-%d')
    test=[]
    max_r=C['max_r']
    count=0
    if include: 
        life_dis=np.zeros((C['max_r']+1))
        print('include')
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf):
                life_dis[C[i][1]]+=1
                count+=1
                test.append(C[i][1])
                count+=1
    else:
        print('exclude')
        life_dis=np.zeros((C['max_r']+1))
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                life_dis[C[i][1]]+=1
                test.append(C[i][1])
                count+=1
    life_dis=life_dis/count
    # Binning 설정 (예: 3개 구간으로 나누기)
    print(1,C['max_r'],len(life_dis))
    num_bins = N
    if not(log):
        bins = np.linspace(0,  C['max_r']+2, num_bins + 1)  # 구간 설정
        x_values=np.arange(C['max_r']+2)
        
        bin_labels = [f"{int(bins[i])} ~ {int(bins[i+1])}" for i in range(len(bins)-1)]

        # 각 bin에 대한 2번 데이터 저장
        binned_data = {label: [] for label in bin_labels}

        # 데이터를 해당 bin에 할당
        for x, y in zip(x_values, life_dis):
            for i in range(len(bins) - 1):
                if bins[i] < x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                    binned_data[bin_labels[i]].append(y)
                    break

        # X축 (bin 범위), Y축 (각 bin의 평균값)
        bins = list(binned_data.keys())  # 범위 (문자열)
        means = [np.mean(np.array(values)) for values in binned_data.values()]  # 각 bin의 평균값 계산


        plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor=c, # 마커 내부 색깔
    markeredgecolor=c,
    lw=3,
    ms=10)
        plt.scatter(axis, dist, marker='o', linestyle='-', color='gray',alpha=0.2)
    #------------------------
    else:
        try:
            data=test.copy()
            min_k=np.min(data)
            max_k=np.max(data)

            dist=np.bincount(data, minlength=C['max_r']+1)/len(data)
            axis=np.arange(0,max_k+1,C['max_r']/N) 
            hist,bins=np.histogram(data,axis,density=True)
            hist=np.zeros(len(hist))
            count=0
            for x in data:
                for i in range(len(axis)-1):
                    if axis[i]<x <= axis[i+1]:
                        hist[i]+=1/(axis[i+1]- axis[i])
                        count+=1
                        break 

            hist=hist/count
            axis2=0.5*(axis[1:]+axis[:-1])
            # print(count)


            plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
        color='black',          # 선 색깔
        marker='o',            # 마커 모양
        markerfacecolor=c, # 마커 내부 색깔
        markeredgecolor=c,
        lw=3,
        ms=10)
            plt.scatter(np.arange(C['max_r']+1), dist, marker='o', linestyle='-', color='gray',alpha=0.2)
        except:
            print(2,len(np.arange(C['max_r']+1)),len(dist))
            print(len(axis2),len(hist))
    # ------------------------
    check=0
    for i in range(len(axis)-1):
        check+=hist[i]*(axis[i+1]-axis[i])
    print(check)
    if not(log):
        plt.yscale('log')
        plt.xscale('log')
    if y_label:
        plt.ylabel("PDF",fontsize=fsize)
    if x_label:
        plt.xlabel("First Rank",fontsize=fsize)
    plt.xticks(fontsize=fsize)  # x축 틱 크기 조절
    plt.yticks(fontsize=fsize)


def last_binning(C ,N,log=False,include=True,x_label=False,y_label=False):
    print('last_binning')
    S=datetime.strptime(C['S'], '%Y-%m-%d')
    E=datetime.strptime(C['E'], '%Y-%m-%d')
    test=[]
    max_r=C['max_r']
    count=0

    if include: 
        life_dis=np.zeros((C['max_r']+1))
        print('include')
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf):
                life_dis[C[i][3]]+=1
                count+=1
                test.append(C[i][3])
                count+=1
    else:

        life_dis=np.zeros((C['max_r']+1))
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])):
                life_dis[C[i][3]]+=1
                test.append(C[i][3])
                count+=1
    life_dis=life_dis/count
    # Binning 설정 (예: 3개 구간으로 나누기)
    num_bins = N
    if not(log):
        bins = np.linspace(0,  C['max_r']+2, num_bins + 1)  # 구간 설정
        x_values=np.arange(C['max_r']+2)
        
        bin_labels = [f"{int(bins[i])} ~ {int(bins[i+1])}" for i in range(len(bins)-1)]

        # 각 bin에 대한 2번 데이터 저장
        binned_data = {label: [] for label in bin_labels}

        # 데이터를 해당 bin에 할당
        for x, y in zip(x_values, life_dis):
            for i in range(len(bins) - 1):
                if bins[i] < x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                    binned_data[bin_labels[i]].append(y)
                    break

        # X축 (bin 범위), Y축 (각 bin의 평균값)
        bins = list(binned_data.keys())  # 범위 (문자열)
        means = [np.mean(np.array(values)) for values in binned_data.values()]  # 각 bin의 평균값 계산


        plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor='red', # 마커 내부 색깔
    markeredgecolor='red',
    lw=3,
    ms=10)
        plt.scatter(axis, dist, marker='o', linestyle='-', color='gray',alpha=0.2)
    #------------------------
    else:
        try:
            data=test.copy()
            min_k=1
            max_k=np.max(data)

            dist=np.bincount(data, minlength=C['max_r']+1)/len(data)
            axis=np.arange(min_k-1,max_k+1,C['max_r']/N) 
            hist,bins=np.histogram(data,axis,density=True)
            hist=np.zeros(len(hist))
            count=0
            for x in data:
                for i in range(len(axis)-1):
                    if axis[i]<x <= axis[i+1]:
                        hist[i]+=1/(axis[i+1]- axis[i])
                        count+=1
                        break 

            hist=hist/count
            axis2=0.5*(axis[1:]+axis[:-1])
            # print(count)

            plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
        color='black',          # 선 색깔
        marker='o',            # 마커 모양
        markerfacecolor='red', # 마커 내부 색깔
        markeredgecolor='red',
        lw=3,
        ms=10)
            plt.scatter(np.arange(C['max_r']+1), dist, marker='o', linestyle='-', color='gray',alpha=0.2)
        except:
            print(2,len(np.arange(C['max_r']+1)),len(dist))
            print(len(axis2),len(hist))
    # ------------------------
    check=0
    for i in range(len(axis)-1):
        check+=hist[i]*(axis[i+1]-axis[i])
    print(check)
    if not(log):
        plt.yscale('log')
        plt.xscale('log')
    if y_label:
        plt.ylabel("PDF",fontsize=fsize)
    if x_label:
        plt.xlabel("Last Rank",fontsize=fsize)
    plt.xticks(fontsize=fsize)  # x축 틱 크기 조절
    plt.yticks(fontsize=fsize)

def last_binning1(C ,N,log=False,include=True,x_label=False,y_label=False):
    print('last_binning1')
    S=datetime.strptime(C['S'], '%Y-%m-%d')
    E=datetime.strptime(C['E'], '%Y-%m-%d')
    test=[]
    max_r=C['max_r']
    count=0

    if include: 
        life_dis=np.zeros((C['max_r']+1))
        print('include')
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf):
                life_dis[C[i][3]]+=1
                count+=1
                test.append(C[i][3])
                count+=1
    else:

        life_dis=np.zeros((C['max_r']+1))
        for i in C.keys()- {'S', 'E','T', 'max_r','max_life'}:
            x=C[i][0] # 수명기간
            y=C[i][1] # 첫 순위
            z=C[i][2] # 맥스 순위
            if (x!=0 and y!=np.inf and z!=np.inf)  and (not(C[i][5]==C['S'] or C[i][6]==C['E'])) :
                life_dis[C[i][3]]+=1
                test.append(C[i][3])
                count+=1
    life_dis=life_dis/count
    # Binning 설정 (예: 3개 구간으로 나누기)
    num_bins = N
    if not(log):
        bins = np.linspace(0,  C['max_r']+2, num_bins + 1)  # 구간 설정
        x_values=np.arange(C['max_r']+2)
        
        bin_labels = [f"{int(bins[i])} ~ {int(bins[i+1])}" for i in range(len(bins)-1)]

        # 각 bin에 대한 2번 데이터 저장
        binned_data = {label: [] for label in bin_labels}

        # 데이터를 해당 bin에 할당
        for x, y in zip(x_values, life_dis):
            for i in range(len(bins) - 1):
                if bins[i] < x <= bins[i + 1]:  # 해당 bin에 속하는 경우
                    binned_data[bin_labels[i]].append(y)
                    break

        # X축 (bin 범위), Y축 (각 bin의 평균값)
        bins = list(binned_data.keys())  # 범위 (문자열)
        means = [np.mean(np.array(values)) for values in binned_data.values()]  # 각 bin의 평균값 계산


        plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
    color='black',          # 선 색깔
    marker='o',            # 마커 모양
    markerfacecolor='red', # 마커 내부 색깔
    markeredgecolor='red',
    lw=3,
    ms=10)
        plt.scatter(axis, dist, marker='o', linestyle='-', color='gray',alpha=0.2)
    #------------------------
    else:
        try:
            data=test.copy()
            min_k=1
            max_k=np.max(data)

            dist=np.bincount(data, minlength=C['max_r']+1)/len(data)
            axis=np.arange(min_k-0.5,max_k+1.5,C['max_r']/N) 
            hist,bins=np.histogram(data,axis,density=True)
            hist=np.zeros(len(hist))
            count=0
            for x in data:
                for i in range(len(axis)-1):
                    if axis[i]<=x < axis[i+1]:
                        hist[i]+=1/(axis[i+1]- axis[i])
                        count+=1
                        break 

            hist=hist/count
            axis2=0.5*(axis[1:]+axis[:-1])
            # print(count)

            plt.plot(axis2, hist,     linestyle='-',         # 선 스타일
        color='black',          # 선 색깔
        marker='o',            # 마커 모양
        markerfacecolor='red', # 마커 내부 색깔
        markeredgecolor='red',
        lw=3,
        ms=10)
            plt.scatter(np.arange(C['max_r']+1), dist, marker='o', linestyle='-', color='gray',alpha=0.2)
        except:
            print(2,len(np.arange(C['max_r']+1)),len(dist))
            print(len(axis2),len(hist))
    # ------------------------
    check=0
    for i in range(len(axis)-1):
        check+=hist[i]*(axis[i+1]-axis[i])
    print(check)
    if not(log):
        plt.yscale('log')
        plt.xscale('log')
    if y_label:
        plt.ylabel("PDF",fontsize=fsize)
    if x_label:
        plt.xlabel("Last Rank",fontsize=fsize)
    plt.xticks(fontsize=fsize)  # x축 틱 크기 조절
    plt.yticks(fontsize=fsize)