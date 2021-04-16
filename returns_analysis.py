#!/usr/bin/env python
# coding: utf-8

# <pre>
# Jupyter Python notebook: returns_analysis
# Version: 0.3.35
# Date: 4/13/2021
# Author: Jarrod Wilcox
# Contact:  jarrod.wilcox@gmail.com
# License: M.I.T. license, distributed as is, suitable for education and research only.
# 
# Oldest installable dependencies tested:  python 3.8.3, numpy 1.15.4, scipy 1.4.1, pandas 1.03, jupyter 1.0.0, cvxpy 1.1.1.
# 
# This "returns_analysis" is research code that first reads a clean comma-delimited file, fully-populated, of asset prices adjusted for any distributions, or alternatively, of their returns.  It is written to be a pre-processor for the "resilience_optimizer" notebook.  Statistical reports, an optional log file, and output files are produced.  A top-down hierarchical clustering algorithm can help spot structural change and outliers.  Scenarios can be extracted from sub-matrices, including those with overlapping compound returns.  Scenarios are described by top-down hierachical clustering, standard statistics including mean, standard deviation, skewness, kurtosis, and correlations.  Return means can be shrunk toward priors that are input from user files.  The generated output files are available as input for the resilience optimizer.
# 
# 
# SAMPLE INPUTS:
# params=collections.OrderedDict(
#     namefile=None,
#     logfile='run013.txt',
#     sourcefile='saved_returns.csv',
#     sourcetype='RETURNS',
#     prior_file='group_priors.csv',
#     codefile='equiv.csv',
#     modify='shrink_means',
#     predict_file="prediction_01.csv",
#     paramfile='params01.json',
#     logging=True,
#     verbose=True,
#     date_format='%m/%d/%y',
#     datebounds=[1980.00,2020.1],
#     scenario_dict={
#         'history':{'multiple':1,'datetimebounds':  ('1/31/88','2/5/21'),'interval':1,'smodify':True,'tdiscount':.03},
#         'breakdown':{'multiple':1,'datetimebounds':('6/30/07','7/31/09'),'interval':1,'smodify':False,'tdiscount':.03}, 
#         '6_month':{'multiple':1,'datetimebounds':('1/31/88','2/5/21'),'interval':6,'smodify':True,'tdiscount':.03},
#         },
#     )
#     
# INPUT DESCRIPTIONS (note file samples may be found in the Github jarrodwilcox/resilience_lab repository):
# 
# 
# namefile:  Optional comma-delimited file (.csv) linking longer asset names to short tickers or other id's.
# 
# logfile:  Path to a log file with all program text output written into it.  Used if logging=True.
# 
# sourcefile:  Path to a csv file containing asset prices or returns, with top row containing tickers or other id's.  The left-most column contains dates (id = Date, date, or DATE) for each joint observation or scenario outcome.  This version of the program is written for monthly data.
# 
# sourcetype: Either the string PRICES or RETURNS.
# 
# prior_file: Path to a csv file containing broad asset return means, standard deviations, and the equivalent number of observations used to estimate them.
# 
# codefile: Path to a csv file associating tickers or other id's to the broad asset classes contained in the prior_file.
# 
# modify: String that controls whether returns in the sourcefile will be modified using the information contained in the codefile and priorfile.  Should be set to either 'shrink_means' or 'mirror'.
# 
# predict_file: Path to an output file in the same format as the sourcefile.  It concatenates return matrices for one or more scenario-generators, ranging from simple history through disaster scenarios through scenarios with returns compounded over multiple periods.  In this version, all scenario generators are derived from the original sourcefile.
# 
# paramfile: Path to a json output file containing these input parameters. 
# 
# logging:  Set either to True or False to determine whether a log file will be written.
# 
# verbose: Set either to True or False to determine how extensive reports will be written.
# 
# date_format: String to govern how Python reads the date column in the sourcefile.
# 
# datebounds: Two element array of floating point dates in years, for easy trimming of the source return data before scenario generation.
# 
# scenario_dict: Python dictionary of dictionaries incorporating descriptions of one or more scenario generators.  The top-level keys are names you supply for each scenario-generator.  Each of thes has subkeys 'multiple', 'datetimebounds','interval','smodify', and 'tdiscount', as described below.
# 
# multiple:  Set at 1 or 0.  This governs whether the current run will generate data for that scenario-generator.
# 
# datetimebounds: Tuple of string-formatted start and stop dates in the same format as in the sourcefile first column (use date_format) that apply to a particular scenario.  The bounds can be wider than available data.
# 
# interval: For original monthly data, use the integer 1.  Higher integers specify that the scenario-generator will supply returns compounded over more than one month. Generated reports for compound returns incorporate measures intended to eliminate the biases resulting from overlapping observations.
# 
# smodify:  A boolean True or False governing whether the return matrix for a particular scenario-generator will be modified.
# 
# tdiscount: A floating point decimal fraction to be preserved in your paramfile dsignating a pure time discounting rate (.03 refers to 3% annually).  This is used when paramfile is read by the resilience_optimizer program.
# 
# Sample data files are included in the repository.
# 
# EXPECTED OUTPUT:
# Produces screen output, an optional logfile with a printout of program input and results, and writes input files for use with the resilience_optimizer program.
#     
# BACKGROUND READING:
# 
# "Better Portfolios with Higher Moments" by Jarrod Wilcox, in the Journal of Asset Management for December,2020, provides further details on the benefits of non-parametric representations of asset return distributions, including their use in constructing more resilient portfolios managing non-normal returns.
# 
# "Expected Surplus Growth Compared with Mean–Variance Optimization", by Jarrod Wilcox and Stephen Satchell in 
# The Journal of Portfolio Management Multi-Asset Special Issue 2021, 47 (4) 145-159; DOI: https://doi.org/10.3905/jpm.2021.1.209 provides further details on the approach.
# 
# </pre>

# DEPENDENCIES

# In[1]:



import sys,collections  
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from math import exp, isnan
import json


# GET PRIORS

# In[2]:


def load_prior_file(prior_file=None):
    if not prior_file is None:
        try:
            prior_file_df=pd.read_csv(prior_file)
            return prior_file_df
        except:
            print('NO PRIOR_FILE FOUND')
            raise


# GET ASSET CODES FOR PRIORS

# In[3]:


# get asset group codes
def load_asset_codes(codefile=None):
    if not codefile is None:
        try:
            equivs_df=pd.read_csv(codefile)
            return(equivs_df)
        except:
            print('NO ASSET CATEGORIZATION FOUND')
            raise


# LOAD DATA FILE

# In[4]:


def load_source(sourcefile):
    try:
        source=pd.read_csv(sourcefile)
        
        clist=source.columns.values #drop unnamed columns
        for i,v in enumerate(clist): 
            if 'named' in v:
                clist[i]= '_'+str(i)        
        source.columns=clist
        for i in clist:
            if i[0]=='_':
                source.drop([i], axis=1, inplace=True)
                
        temp=source.get('Date') #put assumed date column into dataframe index
        if temp is None:
            temp=source.get('DATE')
        if not temp is None:
            source.index=temp 
        try:
            source.drop(['Date'],axis=1,inplace=True)
        except:
            try:
                source.drop(['DATE'],axis=1,inplace=True)
            except:
                pass
        return source
    except:
        print('NO GOOD ' + sourcefile + ' FOUND')
        return None


# CHECK FOR PRIORS

# In[5]:


def check_priors(source_df,prior_file=None,codefile=None):
    print()
    if not (prior_file is None) and not(codefile is None):
        priors_df=load_prior_file(prior_file)
        asset_codes_df=load_asset_codes(codefile)        
        for code in asset_codes_df['asset'].values:
            if not code in priors_df['asset'].values:
                print(code + 'not in '+prior_file)
                raise
        for tick in source_df.columns.values:
            if not tick in asset_codes_df['ticker'].values:
                print(tick + ' not in '+codefile)
                raise
        return (asset_codes_df,priors_df)


# convert date string to year, month

# In[6]:


def convert_datestring(indates,dformat='%Y-%m-%d'):
    dt=[datetime.strptime(d, dformat) for d in indates]
    return [d.year-1+d.month/12. for d in dt]


# CALCULATE RETURNS

# In[8]:


def price2return(prices):
    '''converts monthly prices to monthly returns'''
    returns=prices/prices.shift(1)
    returns=returns[1:]-1.
    returns_df=pd.DataFrame(returns,columns=prices.columns,index=prices.index[1:])   
    return(returns_df)


# MODIFY DATA

# In[9]:


def shrink_means(rtn_df,asset_codes_df,priors_df):

    #prepare ticker: asset codes from possibly larger list to tie to asset priors
    new_asset_codes_df=pd.DataFrame(rtn_df.columns.values,columns=['ticker'])
    asset_dict={x[0]:x[1] for x in asset_codes_df.values}
    new_asset_codes_df['asset']=[asset_dict.get(x) for x in rtn_df.columns.values]
    missing=[v[0] for v in new_asset_codes_df.values if v[1] is None]
    if len(missing)> 0:
        print('The following have no priors associated')
        raise
        
    #calculate observational statistics
    means=np.mean(rtn_df.values,axis=0)
    stds=np.std(rtn_df.values,axis=0)
    stats_df=pd.DataFrame(np.array([means,stds]).T,columns=['o_mean','o_stdev'])
    stats_df['o_N']=len(rtn_df)
    stats_df['ticker']=rtn_df.columns.values
    stats_df['o_precise']=stats_df['o_N']/(stats_df['o_stdev']**2)

    #bring in prior information
    temp_df=new_asset_codes_df.join(priors_df.set_index('asset'),on='asset')
    temp_df['p_precise']=temp_df['p_N']/(temp_df['p_stdev']**2)

    #combine to calulate shrinkage mean
    calc_df=temp_df.join(stats_df.set_index('ticker'),on='ticker')

    calc_df['t_precise']=calc_df['p_precise']+calc_df['o_precise']
    calc_df['wt_p']=calc_df['p_precise']/calc_df['t_precise']
    calc_df['wt_o']=calc_df['o_precise']/calc_df['t_precise']
    calc_df['s_mean']=calc_df['wt_p']*calc_df['p_mean']+calc_df['wt_o']*calc_df['o_mean']

    
    #assemble for display and possible expansion to illustrate tail risk as well
    calc_df=pd.DataFrame(calc_df[['o_mean','s_mean']])
    calc_df['ticker']=stats_df['ticker']
    calc_df=pd.DataFrame(calc_df[['ticker','o_mean','s_mean']])

    #In return matrix, substitute column shrinkage means for column means
    for row,tick in enumerate(calc_df['ticker'].values):
        rtn_df[tick]=rtn_df[tick] + calc_df['s_mean'].values[row] - calc_df['o_mean'].values[row]

    return rtn_df


# In[10]:


def modify_switch(rtn_df,modify, asset_codes_df=None,priors_df=None):
    if modify=='shrink_means':
        return shrink_means(rtn_df,asset_codes_df,priors_df)
    else:
        return rtn_df


# COMPOUND RETURNS OVER RETURN INTERVAL

# In[11]:


def calculate_compound_returns(returns_df,return_interval):
    lrtns_df=np.log1p(returns_df)
    roll_lrtns_df=lrtns_df.rolling(return_interval).sum()[return_interval:]
    returns_df=np.exp(roll_lrtns_df)-1.0
    return returns_df


# COLLECT FURTHER STATISTICS

# In[12]:


def overlap_rtn_summary(returns_df,return_interval):

    tickers=returns_df.columns
    
    for i in range(return_interval):
        rtns=returns_df.values[i:]
        rtns=rtns[::return_interval]
        print('len rtns: ',len(rtns))
        means=np.mean(rtns,axis=0)
        covs=np.cov(rtns.T)
        stdevs=np.std(rtns,axis=0)
        corr2=np.round_(np.corrcoef(rtns.T),2)        
        corrs=np.corrcoef(rtns.T)
        skews=stats.skew(rtns,axis=0)        
        kurts=stats.kurtosis(rtns,axis=0,fisher=False)
        if i==0:
            count=len(rtns)
            means_sum=means
            covs_sum=covs
            stdevs_sum=stdevs
            corrs_sum=corrs
            skews_sum=skews
            kurts_sum=kurts
        else:
            means_sum+=means
            covs_sum+=covs
            stdevs_sum+=stdevs
            corrs_sum+=corrs
            skews_sum+=skews
            kurts_sum+=kurts

    means=np.divide(means_sum,return_interval)
    covs=np.divide(covs_sum,return_interval)
    stdevs=np.divide(stdevs_sum,return_interval)
    corrs=np.divide(corrs_sum,return_interval)
    skews=np.divide(skews_sum,return_interval)
    kurts=np.divide(kurts_sum,return_interval)

    corrs_df=pd.DataFrame(corrs,columns=tickers,index=tickers)
    descript_df=pd.DataFrame({'TICKER':tickers,'MEAN':np.round(means,4),
        'STDEV':np.round(stdevs,4),'SKEW':np.round(skews,2),'KURT':np.round(kurts,2)})
    
    return(descript_df,corrs_df,count) 


# RESEARCH RECORDKEEPING

# In[13]:


def print_parameters2(params):
    for key,value in params.items():
        print("{0}: {1}".format(key,value))
    return


# MATRIX SPLIT

# In[14]:


def split(idxlist,square):
    temp=np.copy(square)
    size=len(square)*.9999    
    while np.linalg.norm(temp,1)<size:
        temp= np.corrcoef(temp.T)
    #now split the list by index and labels
    idx1=[idxlist[i] for i,v in enumerate(temp[0]) if v>0.]
    idx2=[idxlist[i] for i,v in enumerate(temp[0]) if v<= 0.]    
    return (idx1,idx2)


# PREPARE NEW SQUARE SUB-MATRIX FOR SPLITTING

# In[15]:


def squareit(task,master):    
    temp1=np.array([row for i,row in enumerate(master) if i in task])
    temp1T=temp1.T
    temp2=np.array([row for i,row in enumerate(temp1T) if i in task])
    return temp2


# TRAVERSE POTENTIAL CLUSTER TREE

# In[16]:


def traverse(matrix_df):
    labels=matrix_df.columns.values    
    idxs=[i for i,v in enumerate(labels)]
    idx_dict={i:"C" for i in idxs}
    
    master=matrix_df.values

    taskno=0
    tasklist=[idxs]
    while len(tasklist)>taskno:
        task=tasklist[taskno]
        if len(task)>2:
            split_input=squareit(task,master)        
            idx1,idx2=split(task,split_input)       
            for v in idx1:
                idx_dict[v] += '1'
            for v in idx2:
                idx_dict[v] += '2'
            tasklist.append(idx1)
            tasklist.append(idx2)
        taskno+=1
    output1=[[v,idx_dict[i]] for i,v in enumerate(labels)]
    output2=pd.DataFrame(output1,columns=['ID','CLUSTER'])
    output1=output2.sort_values(by=['CLUSTER'])
    return output1


# STATISTICS SUMMARY

# In[17]:


def stat_summary(returns_df,return_interval,verbose=False,namefile=None):
    #CALCULATE RETURN STATISTICS AND SQUARE MATRICES -- COVARIANCE AND CORRELATIONS
    #following improves estimates if overlapping observations
    descript_df,corrs_df,count=overlap_rtn_summary(returns_df,return_interval)
    print(' ')
    print('CLUSTER ANALYSIS')
    cluster_df=traverse(corrs_df) #CLUSTER ANALYSIS
    try:
        if not namefile is None:
            names_df=pd.read_csv(namefile)
            names_dict={x[0]:x[1] for x in zip(names_df['ID'].values,names_df['NAME'])}
            cluster_df['NAME']=[names_dict[x] for x in cluster_df['ID']]
    except:
        print('namefile not located')
        raise
    
    print(cluster_df.to_string(index=False))
    print(' ')
   
    print('DISTRIBUTION STATISTICS')
    descript_df=descript_df.reindex(cluster_df.index)
    print(descript_df.to_string(index=False))
    if verbose:
        print(' ')
        print('CORRELATIONS')
        corrs_df.index=range(len(returns_df.columns)) #needed to put in same indix set as used for reindexing
        corrs_df=corrs_df.reindex(cluster_df.index)
        corrs_df=pd.DataFrame(corrs_df,columns=cluster_df['ID'].values)
        corrs_df.index=corrs_df.columns
        pd.set_option('display.max_rows',None)
        pd.set_option('display.max_columns',None)
        print(corrs_df)
        pd.reset_option("all")
        print(' ')
    

    return (descript_df,corrs_df,count)


# HISTORY ANALYSIS

# In[18]:


def history_analysis(params,data_df):
    
    namefile=params.get('namefile')
    sourcefile=params.get('sourcefile')
    sourcetype=params.get('sourcetype')
    return_interval=params.get('return_interval')
    verbose=params.get('verbose')
    date_format=params.get('date_format')
    datebounds=params.get('datebounds')
    
    #READ IN PRICES OR RETURNS, PROVIDE RETURNS
    try:
        data_df=load_source(sourcefile)
        print('type data_df: ',type(data_df))
    except:
        print("Can't read data.")
        raise
        
    if not (sourcetype=='PRICES' or sourcetype=='RETURNS'):
        print('UNABLE TO DETERMINE SOURCE TYPE')
        raise
        
    if sourcetype=='PRICES':
        print('sourcetype: ',sourcetype)
        returns_df=price2return(data_df)
        returns_df.to_csv('saved_returns.csv')                
    elif sourcetype=='RETURNS':
        returns_df=data_df
    
    #TRIM SAMPLE BY DATE BOUNDS
    if returns_df.index.name in ['Date','DATE','date']:        
        print('datebounds: ',datebounds)        
        print('date_format: ',date_format)
        try:
            returns_df['temp']=[datetime.strptime(d,date_format) for d in returns_df.index.values]
        except:
            print('can not read date as formatted')
            raise
        returns_df['year']=returns_df['temp'].dt.year
        returns_df['month']=returns_df['temp'].dt.month
        returns_df['day']=returns_df['temp'].dt.day
        returns_df['gdate']=round(returns_df['year']-1+ returns_df['month']/12.,2)
        returns_df.drop(['temp','year','month','day'],inplace=True,axis=1)
        gdates=returns_df['gdate'].values
        try:    
            returns_df=returns_df.loc[returns_df['gdate']>=datebounds[0]]
            returns_df=returns_df.loc[returns_df['gdate']<=datebounds[1]]
            returns_df.drop(['gdate'],inplace=True,axis=1)
        except:
            print ('incorrect datebounds!')
            raise
    
    if verbose:

        print(' ')
        print('RETURNS HEAD')
        print(returns_df.head(5))
        print(' ')
        print('RETURNS TAIL')
        print(returns_df.tail(5))
    
    #assume 1 as interval for history
    descript_df,corrs_df,count=stat_summary(returns_df,1,verbose, namefile=None)

    return(returns_df)


# EXTRACT SCENARIO FROM HISTORY

# In[19]:


def data_extract(returns_df,scenario_name,params):
    date_format=params['date_format']
    scenario=params.get('scenario_dict')[scenario_name]
    early,late=scenario['datetimebounds']
    early=datetime.strptime(early,date_format)
    late=datetime.strptime(late,date_format)

    temp_df=returns_df.copy()
    temp_df['temp']=[datetime.strptime(d,date_format) for d in returns_df.index.values]
    temp_df=temp_df.loc[temp_df['temp']>=early]
    temp_df=temp_df.loc[temp_df['temp']<=late]
    temp_df.drop(['temp'],axis=1,inplace=True)
    return temp_df


# SCENARIO ANALYSIS

# In[20]:


def scenario_extract(params,hist_returns_df,asset_codes_df=None,priors_df=None):
    return_interval=params.get('return_interval')  
    verbose=params.get('verbose')
    scenario_dict=params.get('scenario_dict') 
    modify=params.get('modify')
    prediction_sample_length={}    
    prediction_df=None
    
    intervals=[]
    for i,key in enumerate(scenario_dict.keys()):
        properties=scenario_dict[key]
        multiple=int(properties['multiple'])        
        if multiple>=1:
            #SCENARIO PROPERTIES
            print(' ')
            print('SCENARIO')
            scenario_name=key
            print('scenario_name: ',scenario_name)
            print('multiple: ',multiple)
            dates=properties['datetimebounds']
            print('dates: ',dates)
            interval=properties.get('interval')
            if interval is None: #use default if no interval specified for scenario
                interval=1
            print('interval: ',interval)
            intervals.append(interval)
           
            smodify=properties['smodify']
            
            #GET SCENARIO DATA           
            scenario_return_df=data_extract(hist_returns_df,scenario_name,params)
            
            #MODIFICATION -- shrinkage and compounding
            if smodify:
                #note switch is extra hook for future use
                print('MEANS SHRUNK TOWARD ASSET PRIORS')
                scenario_return_df=modify_switch(scenario_return_df,modify,asset_codes_df,priors_df)    
            scenario_return_df=calculate_compound_returns(scenario_return_df,interval)
                
            if verbose:
                print('SCENARIO HEAD')
                print(scenario_return_df.head(5))
                print(' ')
                print('SCENARIO TAIL')
                print(scenario_return_df.tail(5))
                print(' ')
                
            print(' ')
            print('SCENARIO WITH INTERVAL '+str(interval))                   
            descript_df,corrs_df,count=stat_summary(scenario_return_df,interval,verbose, namefile=None)
            
            #MULTIPLY ROWS IF DESIRED (OBSOLETE!) 
            scenario_return_df=pd.concat([scenario_return_df]*multiple,ignore_index=True) #drop dates
            
            #LABEL SCENARIO SOURCE
            scenario_return_df.index=[scenario_name]*len(scenario_return_df) #label scenario source
            
            #APPEND TO PREDICTION DISTRIBUTION LIST
            if prediction_df is None:                
                prediction_df=scenario_return_df.copy()                
            else:           
                prediction_df=pd.concat([prediction_df,scenario_return_df])                
            if not scenario_return_df is None:    
                prediction_sample_length[scenario_name]=len(scenario_return_df)               
            scenario_return_df=None #cleanup for next iteration
            
    print(' ')
    print('SCENARIO SAMPLE LENGTHS')
    for key in prediction_sample_length.keys():
        print(key + ': '+str(prediction_sample_length[key]))
    print(' ')
    test=(float(intervals[0])==sum(intervals)/len(intervals))
    if test:  #if all intervals on same scale
        print('PREDICTION DISTRIBUTION')
        #assume 1 for interval after compounding already
        stat_summary(prediction_df,1,verbose)
    else:
        print('NOTE: PREDICTION DISTRIBUTION INCLUDES DIFFERENT SCALES')

    return prediction_df   


# RETURN MATRIX GENERATOR

# In[21]:


def return_matrix_generator(params={}):    
    namefile=params.get('namefile')
    logfile=params.get('logfile')
    predict_file=params.get('predict_file')
    sourcefile=params.get('sourcefile')
    sourcetype=params.get('sourcetype')    
    logging=params.get('logging')
    verbose=params.get('verbose')
    date_format=params.get('date_format')
    datebounds=params.get('datebounds')
    scenario_dict=params.get('scenario_dict')
    prior_file=params.get('prior_file')
    codefile=params.get('codefile')
    modify=params.get('modify')
    
    # START LOGGING IF SELECTED
    if logging:
        orig_stdout = sys.stdout
        #record results in logfile
        f = open(logfile, 'w')
        sys.stdout = f
    
        #record control parameters
        print_parameters2(params)
    
    # READ NECESSARY FILES
    source_df=load_source(sourcefile)
    print(source_df.head(3))
    asset_codes_df=priors_df=None
    if not modify is None:
        asset_codes_df,priors_df=check_priors(source_df,prior_file,codefile)
   
    # HISTORY ANALYSIS
    returns_df=history_analysis(params,source_df)
    
    # SCENARIO GENERATION, ANALYSIS AND COMPOSITION TO PREDICTION DISTRIBUTION
    if not modify is None: #needs work if multiple options for modify
        prediction_df=scenario_extract(params,returns_df,asset_codes_df,priors_df)
    else:
        prediction_df=scenario_extract(params,returns_df)
    prediction_df.to_csv(predict_file)
    
    # EXIT
    if logging:
        #close logfile and print it on terminal
        f.close()
        sys.stdout = orig_stdout

        h=open(logfile,'r')
        for line in h:         
            print(line[:-1])          
        h.close()
    return


# MAIN: SET PARAMETERS AND CALL RETURN MATRIX GENERATOR

# In[22]:


params=collections.OrderedDict(
    namefile=None,
    logfile='run001A.txt',
    sourcefile='comp_returns.csv',
    prior_file='group_priors.csv',
    codefile='equiv.csv',
    modify='shrink_means',
    predict_file="prediction_01.csv",
    paramfile='params01.json',
    sourcetype='RETURNS',
    logging=True,
    verbose=True,
    
    date_format='%m/%d/%y',
    datebounds=[1980.00,2020.1],
    scenario_dict={
        'history':{'multiple':1,'datetimebounds':('1/31/88','2/5/21'),'interval':1,'smodify':True,'tdiscount':.03},
        'breakdown':{'multiple':1,'datetimebounds':('6/30/07','7/31/09'),'interval':1,'smodify':False,'tdiscount':.03}, 
        '6_month':{'multiple':1,'datetimebounds':('1/31/88','2/5/21'),'interval':6,'smodify':True,'tdiscount':.03},
        },
    )

#run main program
try:    
    dummy=return_matrix_generator(params) 
    paramfile=params.get('paramfile')
    with open(paramfile,'w') as outfile:
        json.dump(params,outfile)
    print(' ')
    print('DONE!')
except:
    print('Unknown Input Error')
    print(' ')
    sys.exit()

     

