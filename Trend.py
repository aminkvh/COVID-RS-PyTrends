import pandas as pd
import numpy as np
import os
from time import sleep
from pytrends.request import TrendReq
import random
import scipy.stats as stats
from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp as mc
from pandas.plotting import table 
plt.style.use('ggplot')
PATH = r"PATH"
ConList= ["",'US','NZ','GB','AU','IE','CA']
Keyw= ['god', 'jesus','prayer',
        'cancer','diabetes','hypertension',
        'cough','fever','sore throat']
keydic = {'rel' : ['god', 'jesus','prayer'],
        'noned' : ['cancer','diabetes','hypertension','cough','fever','sore throat']}

time = ['2015-11-20 2020-07-20']
k=3
col = [['god','jesus','prayer'], 
        ['cancer','diabetes','hypertension'],
        ['cough','fever','sore throat']]

#Get data from google trend
def Gtrends(Keyw, time, ConList):
    
    pytrends = TrendReq(hl='en-US', tz=-360)
    
    for j in time:
        dataset = pd.DataFrame(index=list(pd.date_range(j[0:10], j[11:21]).strftime('%Y-%m-%d')))
        for i in ConList:
                for ii in Keyw:
                    if ii in keydic['rel']:
                        cat = 59
                    else:
                        cat = 0
                    pytrends.build_payload(kw_list=[ii],
                                        cat=cat, 
                                        timeframe=j, 
                                        geo=i)
                    data = pytrends.interest_over_time()
                    if 'isPartial' in data.columns:
                                    data = data.drop(labels=['isPartial'],axis='columns')
                    dataset = dataset.join(data, how='outer')
                dataset.to_csv('5_year_trends__{}__{}.csv'.format(i,j))
                dataset = pd.DataFrame(index=list(pd.date_range(j[0:10], j[11:21]).strftime('%Y-%m-%d')))
    return print('All Done!')

Gtrends(Keyw, time, ConList)

#calculate Cronbach's alpha
def alpha_coren(k, col, df):
    import pandas as pd
    import numpy as np       
    coef = (k/(k-1))
    varcol = np.sum(np.var(df[col]))
    varrow = np.var(np.sum(df[col],axis=1))
    return coef*(1-varcol/varrow)

#Cumulate data
directory = PATH
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.startswith("trend_5_year_trends__"):
            f=open(file, 'r')
            df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df[df.index.week < 26]
            for key in col:
                if 'god' in key:
                   df['rel_sum'] = df[key].sum(axis=1)
                elif 'cancer' in key:
                    df['nonC'] = df[key].sum(axis=1)
                else :
                    df['covidKey'] = df[key].sum(axis=1)
                print(file[21:23] ,'Alpha C for', key , 'is :', alpha_coren(3, key, df))
            c = ['rel_sum_2016','nonC_2016','covidKey_2016','rel_sum_2017','nonC_2017','covidKey_2017','rel_sum_2018','nonC_2018','covidKey_2018','rel_sum_2019','nonC_2019','covidKey_2019','rel_sum_2020','nonC_2020','covidKey_2020']
            df = df.iloc[:,-3:]
            df = pd.concat([df.iloc[i:i+25].reset_index(drop=True) for i in range(0,125,25)],axis= 1)
            df.columns = c
            c.sort()
            df = df[c]
            df.to_csv('trend_{}.csv'.format(file[20:23]))
            f.close()

#Correlation Heatmap
directory = PATH
os.chdir(directory)
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            f=open(file, 'r')
            if file == "mt19937-testset-1.csv": break
            else:
                pass
            df = pd.read_csv(f)
            df.columns = ['rel 16','rel 17','rel 18','rel 19','rel 20',
                            'nonc 16','nonc 17','nonc 18','nonc 19','nonc 20',
                            'des 16','des 17','des 18','des 19','des 20']
            dicm = {'rel 16': [],
                    'rel 17': [],
                    'rel 18': [],
                    'rel 19': [],
                    'rel 20': []}
            for i in df.columns[:5]:
                dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+5][:len(df.iloc[:, df.columns.get_loc(i)+5])-2])[0])
            for i in df.columns[:5]:
                dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+10][:len(df.iloc[:, df.columns.get_loc(i)+10])-2])[0])
            for i in df.columns[:5]:
                dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+5][:len(df.iloc[:, df.columns.get_loc(i)+5])-2])[1])
            for i in df.columns[:5]:
                dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+10][:len(df.iloc[:, df.columns.get_loc(i)+10])-2])[1])           
            rf = pd.DataFrame.from_dict(dicm)
            rf.index = ['NCD', 'Covid', 'SigN', 'SigCovid']
            rf.columns = ['2016', '2017', '2018', '2019', '2020']
            rf = rf.round(3)
            l = [[str(p) + '\n' + '(' + str(s) + ')' for p, s in zip(rf.iloc[0], rf.iloc[2])],
                [str(p) + '\n' + '(' + str(s) + ')' for p, s in zip(rf.iloc[1], rf.iloc[3])]]
            ax = plt.axes()
            plot = sns.heatmap(rf.iloc[:2,:], cbar = False , square=True, annot=l ,fmt='', linewidths= 1)
            ax.set_title('{}'.format(file[0:2]))
            plot.figure.savefig('G-{}.png'.format(file[0:2]))
            plt.clf()
            f.close()

#anova
directory = PATH
os.chdir(directory)
p = []
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            f=open(file, 'r')
            df = pd.read_csv(f)
            df.columns = ['rel 16','rel 17','rel 18','rel 19','rel 20',
                            'nonc 16','nonc 17','nonc 18','nonc 19','nonc 20',
                            'des 16','des 17','des 18','des 19','des 20']
            a = list(stats.f_oneway(df['rel 16'],df['rel 17'],df['rel 18'],df['rel 19'],df['rel 20']))
            a1 = list(stats.f_oneway(df['nonc 16'],df['nonc 17'],df['nonc 18'],df['nonc 19'],df['nonc 20']))
            a2 = list(stats.f_oneway(df['des 16'],df['des 17'],df['des 18'],df['des 19'],df['des 20']))
            conall = a + a1 + a2
            p.append(conall)
            f.close()
    cc= ['rel', 'rel_sig', 'NCD', 'NCD_sig', 'COVID', 'COVID_sig']
    indx = ['AU', 'CA', 'GB', 'IE', 'NZ', 'US']
    dff = pd.DataFrame(p, index= indx, columns=cc).round(5)
    dff.to_csv('anova.csv')
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    ax.set_frame_on(False) 
    tabla = table(ax, dff, loc='upper right', colWidths=[0.17]*len(df.columns))
    tabla.auto_set_font_size(False) 
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.2)
    plt.savefig('table.png', bbox_inches='tight')

#alpha c
directory = PATH
os.chdir(directory)
a = pd.DataFrame()
for root,dirs,files in os.walk(directory):
    for file in files:
        p=[]
        if file.startswith("5"):
            f=open(file, 'r')
            df = pd.read_csv(f)
            for i in col:
                p.append(alpha_coren(k, i, df.iloc[-30:-1]).round(3))
            c = pd.DataFrame([p], columns=['rel','NCD','COVID'], index= [file[1:3]])
            f.close()
        a = pd.concat([a,c])
    a = a.T
    a = a.iloc[:,:6]
    a.to_csv('Alpha_cor.csv')
    ax = plt.axes()
    o = pd.DataFrame()
    for i in a:
        for j in a.index:
            if a.loc[j, i] > 0.7:
                o.loc[j, i] = str(a.loc[j, i]) + '*'
            else:
                o.loc[j, i] = str(a.loc[j, i])
    sns.heatmap(a, square=(2,2),linewidths= 1, annot=o, fmt= "", cbar=False)
    ax.set_title('Alpha Cronbach')
    plt.yticks(va="center")
    plt.savefig('Alpha Cronbach')
    plt.clf()
    from pandas.plotting import table        
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    ax.set_frame_on(False) 
    tabla = table(ax, o, loc='upper right', colWidths=[0.17]*len(df.columns))
    tabla.auto_set_font_size(False) 
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.2)
    plt.savefig('table.png', bbox_inches='tight')

#cum time/key plot
directory = PATH
os.chdir(directory)
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            f = open(file, 'r')  
            df = pd.read_csv(f)            
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(6,6))
            labels = ['2016', '2017', '2018', '2019', '2020']
            ax1.plot(df.iloc[:,10:15])
            ax2.plot(df.iloc[:,0:5])
            ax3.plot(df.iloc[:,5:10])
            ax1.set_title('I - COVID', size =10)
            ax2.set_title('II - Religion/Spirituality', size =10)
            ax3.set_title('III - NCD', size =10)
            fig.text(0.5, 0.13, 'Time (Week)', ha='center')
            fig.text(-0.03, 0.5, 'Cumulative Search Volume', va='center', rotation='vertical')
            fig.text(0.45, 1, '{} Search '.format(file[0:2]), va='center')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9 ])
            plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5,-0.7), ncol=5)
            plt.tight_layout()
            plt.savefig('CUM-{}.png'.format(file[0:2]), bbox_inches='tight')
            plt.clf()
            f.close()

# Relational plot
directory = PATH
os.chdir(directory)
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            f = open(file, 'r')  
            df = pd.read_csv(f)
            for i in range(5):
                sns.regplot(df.iloc[2:,i], df.iloc[:len(df.index)-2,i+10], 
                            ci=False, scatter=False, 
                            label='20{}'.format(df.columns[i][-2:]))
            plt.legend()
            plt.title('{} R/S to COVID Search'.format(file[:2]))
            plt.xlabel('R/S Cumulative Search')
            plt.ylabel('COVID Cumulative Search')
            plt.savefig('rs-{}.png'.format(file[0:2]), bbox_inches='tight')
            plt.clf()
            '''for i in range(5):
                sns.regplot(df.iloc[2:,i], df.iloc[:len(df.index)-2,i+5], 
                            ci=False, scatter=False, 
                            label='20{}'.format(df.columns[i][-2:]))
            plt.legend()
            plt.title('{} NCD to COVID Search'.format(file[:2]))
            plt.xlabel('NCD Cumulative Search')
            plt.ylabel('COVID Cumulative Search')
            plt.savefig('NCD-{}.png'.format(file[0:2]), bbox_inches='tight')
            plt.clf()
            f.close()'''

#Anova Box plot
directory = PATH
os.chdir(directory)
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            f = open(file, 'r')  
            df = pd.read_csv(f)
            df.columns = ['R/S 16','R/S 17','R/S 18','R/S 19','R/S 20',
                        'NCD 16','NCD 17','NCD 18','NCD 19','NCD 20',
                        'COVID 16','COVID 17','COVID 18','COVID 19','COVID 20']            
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6,6))
            sns.boxplot(x="variable", y="value", data=pd.melt(df.iloc[:,:5]), ax=ax1)
            sns.boxplot(x="variable", y="value", data=pd.melt(df.iloc[:,5:10]), ax=ax2)
            sns.boxplot(x="variable", y="value", data=pd.melt(df.iloc[:,10:15]), ax=ax3)
            ax1.set_ylabel('')    
            ax1.set_xlabel('')
            ax2.set_ylabel('')    
            ax2.set_xlabel('')
            ax3.set_ylabel('')    
            ax3.set_xlabel('')
            ax1.set_title('I - Religion/Spirituality', size =10)
            ax2.set_title('II - NCD', size =10)
            ax3.set_title('III - COVID-19', size =10)
            fig.text(0.35, 1, '{} Search Volum For Each Set'.format(file[0:2]), va='center')
            plt.tight_layout()
            plt.savefig('BOX-{}.png'.format(file[0:2]), bbox_inches='tight')
            plt.clf()
        f.close()

#WW Data
df = pd.read_csv('WW.csv')
c= sorted(list(sum([('1R/S'+str(i), 'NCD' + str(i), 'COVID' + str(i)) for i in range(16,21)], ())))
df = df[c]
dicm = {'1R/S16': [],
        '1R/S17': [],
        '1R/S18': [],
        '1R/S19': [],
        '1R/S20': []}
for i in df.columns[:5]:
    dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+5][:len(df.iloc[:, df.columns.get_loc(i)+5])-2])[0])
for i in df.columns[:5]:
    dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+10][:len(df.iloc[:, df.columns.get_loc(i)+10])-2])[0])
for i in df.columns[:5]:
    dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+5][:len(df.iloc[:, df.columns.get_loc(i)+5])-2])[1])
for i in df.columns[:5]:
    dicm[i].append(pearsonr(df[i][2:],df.iloc[:, df.columns.get_loc(i)+10][:len(df.iloc[:, df.columns.get_loc(i)+10])-2])[1])           
rf = pd.DataFrame.from_dict(dicm)
rf.index = ['Covid', 'NCD', 'SigN', 'SigCovid']
rf.columns = ['2016', '2017', '2018', '2019', '2020']
            # rf.loc[['SigN', 'SigCovid']] =rf.loc[['SigN', 'SigCovid']]/2
rf = rf.round(3)
l = [[str(p) + '\n' + '(' + str(s) + ')' for p, s in zip(rf.iloc[0], rf.iloc[2])],
    [str(p) + '\n' + '(' + str(s) + ')' for p, s in zip(rf.iloc[1], rf.iloc[3])]]
ax = plt.axes()
plot = sns.heatmap(rf.iloc[:2,:], cbar = False , square=True, annot=l ,fmt='', linewidths= 1)
ax.set_title('World')
plot.figure.savefig('G-WW.png')
plt.clf()

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(6,6))
labels = ['2016', '2017', '2018', '2019', '2020']
ax3.plot(df.iloc[:,10:15])
ax2.plot(df.iloc[:,0:5])
ax1.plot(df.iloc[:,5:10])
ax1.set_title('COVID', size =10)
ax2.set_title('Religion/Spirituality', size =10)
ax3.set_title('NCD', size =10)
fig.text(0.5, 0.13, 'Time (Week)', ha='center')
fig.text(-0.03, 0.5, 'Cumulative Search Volume', va='center', rotation='vertical')
fig.text(0.45, 1, 'WW Search ', va='center')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9 ])
plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5,-0.7), ncol=5)
plt.tight_layout()
plt.savefig('CUM-WW.png', bbox_inches='tight')
plt.clf()

for i in range(5):
    sns.regplot(df.iloc[2:,i], df.iloc[:len(df.index)-2,i+5], 
                ci=False, scatter=False, 
                label='20{}'.format(df.columns[i][-2:]))
plt.legend()
plt.title('WW R/S to COVID Search')
plt.xlabel('R/S Cumulative Search')
plt.ylabel('COVID Cumulative Search')
plt.savefig('R-S-WW.png', bbox_inches='tight')
plt.clf()


df = pd.read_csv(PATH)
df['Month'] = pd.to_datetime(df['Month'])
plt.plot(df['Month'], df['Religion: (Worldwide)'], marker='.', linestyle='none')