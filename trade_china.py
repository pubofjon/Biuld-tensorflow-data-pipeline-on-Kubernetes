import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime, time
#from ML_2017Feb19_PROD import ml_knn, ml_knn_apply

def import_tick():
    df=pd.read_excel(r'C:\pycode\pytest\records.xlsx', sheetname="tick")
    df['date']=pd.to_datetime(df['date'], format='%Y%m%d')
    dtd=pd.DataFrame()
    dtd['open']=df.groupby(df['date']).first()['open']
    dtd['close']=df.groupby(df['date']).last()['close']
    dtd['low']=df.groupby(df['date'])['high'].max()
    dtd['high']=df.groupby(df['date'])['high'].max()
    dtd['low']=df.groupby(df['date'])['low'].max()
    dtd['volume']=df.groupby(df['date'])['volume'].sum()
    dtd['amt']=df.groupby(df['date'])['amt'].sum()
    dtd['p_pct_chg']=dtd['close']/dtd['close'].shift(1)-1
    dtd['v_pct_chg']=dtd['volume']/dtd['volume'].shift(1)-1
    dtd['atr_pct']=(dtd['high']-dtd['low'])/dtd['close'].shift(1)
    #dtd.drop(dtd.index[[0,0]], inplace=True)
    dtd['date']=df.groupby(df['date'])
    return dtd,df

def import_data():
    dfs=pd.DataFrame()
    df=pd.read_excel(r'C:\pycode\pytest\records.xlsx', sheetname="data")
    df['pl']=-df['amt_dealt']*df['bss']
    df['position']=df['qty_dealt']*df['bss']
    outlier_position=['2018-04-19','2018-04-20','2017-11-16','2017-11-17']
    con_pos=df['date'].isin(outlier_position)
    df=df[~con_pos]
    df['datetime']=df['date'].astype(str)+ " "+ df['timestamp'].astype(str)
    df['datetime']=pd.to_datetime(df['datetime'])
    df['dts']=df['datetime'].dt.round('min')
    dfs['pl']=df.groupby(df['date'])['pl'].sum()
#    position_by_date=df.groupby(df['date'])['position'].sum()
    return df, dfs

def anomaly_check(df):
        print("total trade count:", df.shape[0])
        df['price_dealt_pctl']=(df['price_dealt']-df['low'])/(df['high']-df['low'])
        df[np.isinf(df['price_dealt_pctl'])]=np.nan
        df[np.isneginf(df['price_dealt_pctl'])]=np.nan
        inf_count=df[np.isnan(df['price_dealt_pctl'])].shape[0]
        print("inf_count:", inf_count)
#        df.dropna(inplace=True)
        con_odd=(df['price_dealt_pctl']<0) | (df['price_dealt_pctl']>1)
        con_b_odd=(df['bss']==1) & con_odd
        con_s_odd=(df['bss']==(-1)) &  con_odd
        b_pctl_odd=df[con_b_odd][['date','timestamp','price_dealt', 'bs', 'high','low']]
        s_pctl_odd=df[con_s_odd][['date','timestamp','price_dealt','bs', 'high','low']]
        print("number of odd price:", df[con_odd].shape[0])
        print("number of normal price:", df[~con_odd].shape[0])
        print("number of buy price anamoly:  ", b_pctl_odd.shape[0])
        print("number of sell price anamoly:  ", s_pctl_odd.shape[0])
#        print(b_pctl_odd)
#        print(s_pctl_odd)
        con_b=(df['bss']==1) & (~con_odd)
        con_s=(df['bss']==(-1)) & (~con_odd)
        b_pctl=df[con_b]
        s_pctl=df[con_s]
        print("b_pctl stat: \n", b_pctl['price_dealt_pctl'].describe())
        b_pctl['price_dealt_pctl'].plot.hist(bins=10, alpha=0.2)
        print("s_pctl stat: \n ", s_pctl['price_dealt_pctl'].describe())
        s_pctl['price_dealt_pctl'].plot.hist(bins=10, alpha=0.2)
        plt.show()
        return dt

def prep_data(df, dt): #df: trade, dt:tick, to prepare traiing and testing dataset
    dt=dt[['datetime', 'open', 'high', 'low', 'close',
       'volume', 'amt', 'chg', 'pct_chg']]
    dt['dts']=dt['datetime']
    dt=dt.drop(['datetime'], axis=1)
    dt=dt.set_index('dts')
    df=df[['dts', 'timestamp','bs','bss','pl','position','price_dealt','qty_dealt', 'amt_dealt', 'qty_recall']]
    df=df.set_index('dts')
    dft=df.join(dt, how='outer')    
    d1=dft
    d1['atr']=(d1['high']-d1['low'])/d1['open']   #ATR: average true range - volatility 
    d1['qty_bs']=d1['qty_dealt']*d1['bss'] #qty_bs: quanty bought or sell - as ML's target 
    d1['pos_bal']=d1['position'].cumsum()  # position_balance - cumulated positions at each tick
    d0=d1[['open','volume','atr','pos_bal', 'qty_bs']]  # 4 featuers, 1 targets for ML
    d0.fillna(0,inplace=True)
    dtrain=d0.iloc[:24000,:]  #define training set
    dtest=d0.iloc[24000:,:] #define testing set (20%)
    return dtrain, dtest
    


def try_knn(dtrain, dtest, dtrade): #run ML testing script

    ml_knn(dtrain)  #run ML and save ML model
    y=ml_knn_apply(dtest)  #apply ML model to testing data
    dy=dtrade.iloc[24000:,:]  #
    dy['pred']=y
    dy['pred'].cumsum().plot()   #predicted qty_bs cumulated position as a check point

def ml_knn(df): #knn requires target be lable (not numeric)
    from sklearn import neighbors
    
    clf=neighbors.KNeighborsClassifier(n_neighbors=3)  #n=3
    df.sort_index(axis=1)
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,df.shape[1]-1]
#dataframe indexing: df[train_index] will return columns labels rather than row indices
#ndarray indexing: collection of row indicies
#so either use .iloc accessor or conver dataframe to array, x=x.values
    x=x.values 
    y=y.values
    
    from sklearn.cross_validation import KFold
    scores=[]
#    kf=KFold(n=len(x), n_folds=5, shuffle=False, random_state=None)
    kf=KFold(n=x.__len__(), n_folds=5, shuffle=False, random_state=None)
    for train_index, test_index in kf:
        x_train, y_train=x[train_index],y[train_index]
        #print("train:", train_index, "test:", test_index)
        x_test, y_test=x[test_index], y[test_index]
        clf.fit(x,y)
        scores.append(clf.score(x_test, y_test))
    print ("mean (scores) =%.5f\t Stddev(scores)=%.5f"%(np.mean(scores),np.std(scores)))
# dave the model
    pickle.dump(clf, open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model.dat',"wb"))
    #clf_1=pickle.load(open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model.dat',"r") )
#ml_knn_model_1.dat:  ['hi_5y', 'hi_10y', 'hi_22y', 'hi_66y', 'lo_5y','lo_10y', 'lo_22y',\
#     'lo_66y']
    
def ml_knn_apply(df):
    from sklearn import neighbors
    df.sort_index(axis=1)
    x=df.iloc[:,:-1]
    y=df.iloc[:,df.shape[1]-1]
    x=x.values 
    y=y.values
    clf=pickle.load(open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model.dat',"rb"))
    y_pred=clf.predict(x)
    #error=y-y_pred
   # dc=pd.DataFrame([y_pred, y], columns=['y_pred','y'])
    ar=np.concatenate((y_pred, y), axis=0)  
    t=y_pred==y
    unique, counts = np.unique(t, return_counts=True)

#    df['pred']=y_pred
    print("false/true: ", counts)
    return y_pred

dtd, dt=import_tick()
df,dfs=import_data()    
#anomaly_check(df)     
dtrain, dtest=prep_data(df,dt)
#pl_by_date.cumsum().plot()
try_knn(dtrain, dtest, df)