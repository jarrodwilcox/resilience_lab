#!/usr/bin/env python
# coding: utf-8

# <pre>
# Intended Module: resilience_optimizer
# Author: Jarrod Wilcox
# Version: 0.6
# Date: 4/13/2021
# Contact:  jarrod.wilcox@gmail.com
# 
# Resilience_optimizer is a basic research program used subsequent to preprocessing by the returns_analysis program to produce more resilient portfolio allocations.  It is based on a combination of user_supplied weights for multiple scenarios, including history and scenarios in which input returns are compounded over multiple periods.  Investment return probabilities are combined with maximization of expected Rubinstein utility ln(1+Lr) to produce more resilient portfolios with less tail risk.  If all the scenarios are based on the same time horizon, a comparison with allocations based on Markowitz's mean-variance approach is also reported.
# 
# Distributed as is, with MIT license, suitable for education and research only.
# 
# Oldest installable dependencies tested:  python 3.8.3, numpy 1.15.4, scipy 1.4.1, pandas 1.03, jupyter 1.0.0, cvxpy 1.1.1.
# 
# SAMPLE INPUTS:
# params=dict(
#     journalfile='JOURNAL.txt',
#     logfile='run001.txt',
#     sample='GITHUB EXAMPLE',
#     sourcefile='prediction_01.csv',
#     in_paramfile='params01.json',
#     Llist=[1,2,4,8,16],
#     worst=(-0.99),
#     logging=True,
#     verbose=True,
#     uweights={'history':[0.4],'breakdown':[0.2],'6_month':[0.4]},
#     )
# 
# journalfile: Path to file generated to append a list of run input descriptors that can be used as a directory for various optimization run log files.
# 
# logfile: Path to file that mirrors the printed program output for this set of inputs
# 
# sample: String describing the run to jog the researcher's memory
# 
# sourcefile: Path to a comma-delimited return file with the leftmost column labeling a scenario source and remaining column headers identifying the security (usually a ticker). WARNING:  Only sourcefiles produced by the returns_analysis preprocessor should be used.
# 
# in_paramfile: Path to parameter file produced by preprocessor returns_analysis.
#     
# Llist: a vector of risk aversion coefficients, with larger numbers representing more conservatism.
# 
# worst: -0.99:  Negative floating point number between 0.0 and -1.0. If the solution is feasible, this constrains candidate allocations to scenario returns so that the optimizer cannot explore impossible long-only in estimating the objective function gradient.  This is intended to prevent premature raising of errors.
# 
# logging: Boolean to govern whether the log file is to be created.
# 
# verbose: Boolean to provide additional reporting of tail risk potential of the output allocation
# 
# uweights: Dictionary of scenario-generator labels and weights to be given to different scenario generators.  The labels must correspond to those transmitted by the returns_analysis program with positive multiple (1).
# 
# EXPECTED OUTPUT:
# Produces screen output and an optional logfile with a printout of program input and results.
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


import sys,json
import numpy as np
import pandas as pd
from scipy import stats
import cvxpy as cp    


# LOAD DATA FILE

# In[2]:


def load_source(sourcefile): 
    try:
        source=pd.read_csv(sourcefile)            
        return source
    except:
        print('NO ' + sourcefile + ' FOUND')
        return None


# CVXPY OPTIMIZER

# In[3]:


def optim_with_cvxpy2(rtns,levs,mns,mcovs,headers,labels,worst,log_tdiscount,row_weight,ob):

    barrier=worst+1
    merit=pd.DataFrame(index=headers,columns=labels)
    nrows,ncols=rtns.shape
    nlevs=len(levs)
    alloc=np.ones((nlevs,ncols),dtype='float64')
    prtns=np.zeros((nlevs,nrows),dtype='float64')
    
    xx=cp.Variable(ncols)
    for i in range(nlevs):
        lev=levs[i]
        levreturn=(rtns*lev)
        print("Risk Aversion: ",lev)
        
        if ob=='MV':           
            constraints =[sum(xx)==1, 0<=xx, xx<=1] #Long-only portfolios                   
            objective=cp.Minimize(-cp.sum(cp.multiply(mns,xx)) + lev*cp.quad_form(xx,mcovs)/2.0)
            prob=cp.Problem(objective,constraints)
            result=prob.solve(eps_abs=1e-7,eps_rel=1e-7)
            xxvalue=xx.value
            prtns[i]=np.dot(rtns,xxvalue)
            merit['M_objective'][headers[i]]= np.sum(mns*xxvalue) - levs[i]*np.dot(np.dot(xxvalue,mcovs),xxvalue.T)/2.0
            merit['W_objective'][headers[i]]= np.sum(np.multiply(row_weight,(np.log1p(np.dot(levreturn,xxvalue.T))+log_tdiscount)))   

        elif ob=='LLS':
            constraints =[sum(xx)==1, 0<=xx, xx<=1, -1.0+barrier <= levreturn @ xx ] #Long-only portfolios                              
            objective=cp.Maximize(cp.sum(cp.multiply(row_weight,cp.log1p(levreturn @ xx)+log_tdiscount)))
            prob=cp.Problem(objective,constraints)
            result=prob.solve(abstol=1e-7,reltol=1e-7,verbose=False)/nrows
            xxvalue=xx.value
            if xxvalue is None:                
                print('WARNING!!!! cvxpy problem appears not feasible.')
                raise
            prtns[i]=np.dot(rtns,xxvalue)
            merit['M_objective'][headers[i]]= sum(mns*xxvalue) - levs[i]*np.dot(np.dot(xxvalue,mcovs),xxvalue.T)/2.0
            merit['W_objective'][headers[i]]= np.sum(np.multiply(row_weight,(np.log1p(np.dot(levreturn,xxvalue.T))+log_tdiscount)))       
        alloc[i]=xxvalue 
        merit['norm2'][headers[i]] = np.dot(xxvalue,xxvalue)        
    
    return (prtns[::].T,alloc[::],pd.DataFrame.copy(merit,deep=True))


# PRINT ALLOCATION RESULTS

# In[4]:


def print_alloc(alloc_df):   
    ilist=[]
    rowlist=[]
    for i,row in enumerate(alloc_df.values):
        for v in row:
            if abs(v)>.00001:
                ilist.append(alloc_df.index[i])
                rowlist.append([str(x) for x in row])                
                break
    chosen_df=pd.DataFrame(rowlist,index=ilist,columns=alloc_df.columns)       
    print(chosen_df)  


# FIND BEST ALLOCATION

# In[5]:


def find_best_allocation(rtns_df,worst,levs,meta_df=False,valid_mv=None,verbose=True): 
    rtns_df.to_csv('temp1.csv')
    rtns=rtns_df.values    
    rtns_cols=rtns_df.columns
      
    means=np.mean(rtns_df,axis=0)
    covs=np.cov(rtns_df.T)
    
    mpreturns=None
    malloc=None
    M_merit=None
    
    log_tdiscount=0
    row_weight=1
    if not meta_df is None:
        log_tdiscount=meta_df['log_tdiscount'].values
        row_weight=meta_df['row_weight'].values  
    
    headers=['L:'+x for x in list(map(str,levs))]
    labels=['W_objective','M_objective','norm2']

    if verbose and valid_mv:
        print(' ')    
        print('RUNNING MEAN-VARIANCE OPTIMIZATION')
    if valid_mv:
        mpreturns,malloc,M_merit=optim_with_cvxpy2(rtns,
            levs,means,covs,headers,labels,worst,log_tdiscount,row_weight,ob="MV")
    if verbose:
        print(' ')
        print('RUNNING EXPECTED SURPLUS GROWTH OPTIMIZATION')
    wpreturns,walloc,W_merit=optim_with_cvxpy2(rtns,
        levs,means,covs,headers,labels,worst,log_tdiscount,row_weight,ob="LLS")
    
    mpreturns_df=pd.DataFrame(mpreturns,index=rtns_df.index)
    wpreturns_df=pd.DataFrame(wpreturns,index=rtns_df.index)
    
    if verbose:
        if valid_mv:
            print(' ')
            print('ALLOCATIONS TO MAXIMIZE MEAN-VARIANCE')  
            malloc_df=pd.DataFrame(np.round(malloc,3),columns=rtns_df.columns,index=headers).T
            print_alloc(malloc_df)
        print(' ')
        print('ALLOCATIONS TO MAXIMIZE EXPECTED SURPLUS GROWTH')
        walloc_df=pd.DataFrame(np.round(walloc,3),columns=rtns_df.columns,index=headers).T
        print_alloc(walloc_df)
        print(' ')
        if valid_mv:
            print('IN-SAMPLE ALLOCATION MERIT FROM MEAN-VARIANCE')
            print(M_merit[['W_objective','M_objective','norm2']].head(10))
            print(' ')    
        print('IN-SAMPLE ALLOCATION MERIT FROM EXPECTED SURPLUS GROWTH')
        print(W_merit[['W_objective','M_objective','norm2']].head(10))
        print(' ')
    
    return (wpreturns_df,mpreturns_df,walloc,malloc,W_merit,M_merit)


# DESCRIBE RETURN DISTRIBUTION

# In[6]:


def rtn_summary(returns_df):
    tickers=returns_df.columns   
    rtns=returns_df.values
    print('len rtns: ',len(rtns))
    means=np.mean(rtns,axis=0)
    covs=np.cov(rtns.T)
    stdevs=np.std(rtns,axis=0)        
    corrs=np.corrcoef(rtns.T)
    skews=stats.skew(rtns,axis=0)        
    kurts=stats.kurtosis(rtns,axis=0,fisher=False)
    
    corrs_df=pd.DataFrame(corrs,columns=tickers,index=tickers)
    descript_df=pd.DataFrame({'TICKER':tickers,'MEAN':np.round(means,4),
        'STDEV':np.round(stdevs,4),'SKEW':np.round(skews,2),'KURT':np.round(kurts,2)})
    
    return(descript_df,corrs_df,count) 


# EXTEND ALLOCATION CONSEQUENCES TO PORTFOLIO RETURN CHARACTERISTICS IN SAMPLE

# In[7]:


def show_xray(merit,objective,levs,pmean,pstd,pskew,pkurt):
    #X-ray on optimal in-sample surplus log growth rate objective
    exp_utility=[x for x in merit[objective].values]
    xray=pd.DataFrame([levs,exp_utility,pmean,pstd,pskew,pkurt]).T
    xray.columns=['Leverage','Exp_Log_Gr','mean','stdev','skewness','kurtosis']
    
    xray['Q'] = xray['Leverage']*xray['stdev']/(1+xray['Leverage']*xray['mean'])
    headers=['L:'+x for x in list(map(str,levs))]
    Q_df=pd.DataFrame([headers,xray['Q']]).T
    Q_df.columns=['Leverage','Q']
    with pd.option_context('display.float_format', '{:,.3f}'.format):
        print(Q_df.to_string(index=False))
    
    xray['First']= np.log1p(xray['Leverage']*xray['mean'])
    xray['Second']=-(xray['Q']**2)/2
    xray['Third']=xray['skewness']*(xray['Q']**3)/3
    xray['Fourth']=-xray['kurtosis']*(xray['Q']**4)/4
    xray['Residual']=xray['Exp_Log_Gr']-xray['First']-xray['Second']-xray['Third']-xray['Fourth']
    xray=xray.drop(['mean','stdev','skewness','kurtosis','Q'],axis=1)
    print(' ')
    print('COMPOSITION BY RETURN DISTRIBUTION MOMENTS:')
    print(' ')
    print(xray.to_string(index=False))    
    
    return None


# In[8]:


def describe_portfolio_returns(wprtns,mprtns,wmerit,mmerit,levs,verbose=True,valid_mv=None):
    #COMPARE OUTPUTS ON TRADITIONAL STATISTICS
    print(' ')
    if verbose:
        wpmean=pd.Series(np.mean(wprtns,axis=0))
        wpstd=pd.Series(np.std(wprtns,axis=0))
        wpskew=pd.Series(stats.skew(wprtns,axis=0))
        wpkurt=pd.Series(stats.kurtosis(wprtns,axis=0,fisher=False))    

        mpmean=pd.Series(np.mean(mprtns,axis=0))
        mpstd=pd.Series(np.std(mprtns,axis=0))
        mpskew=pd.Series(stats.skew(mprtns,axis=0))
        mpkurt=pd.Series(stats.kurtosis(mprtns,axis=0,fisher=False))
    
        pdescribe1=pd.DataFrame({'WMEAN': np.round(wpmean,4),'MMEAN':np.round(mpmean,4)})
        pdescribe2=pd.DataFrame({'WSTD': np.round(wpstd,4),'MSTD':np.round(mpstd,4)})
        pdescribe3=pd.DataFrame({'WSKEW': np.round(wpskew,3),'MSKEW':np.round(mpskew,3)})
        pdescribe4=pd.DataFrame({'WKURT': np.round(wpkurt,3),'MKURT':np.round(mpkurt,3)})   
        pdescribe=pd.concat([pdescribe1,pdescribe2,pdescribe3,pdescribe4],axis=1,sort=False)
        pdescribe.index=['L:'+x for x in list(map(str,levs))]
        print('COMPARE PORTFOLIO STATISTICS')
        print(pdescribe)
        print(' ')
        
        if verbose and valid_mv:
            print('IN_SAMPLE SURPLUS GROWTH OBJECTIVE WITH MEAN-VARIANCE OPTIMIZATION')
            show_xray(mmerit,'W_objective',levs,mpmean,mpstd,mpskew,mpkurt)
            print(' ')
        print('IN-SAMPLE SURPLUS GROWTH OBJECTIVE WITH SURPLUS GROWTH OPTIMIZATION')
        show_xray(wmerit,'W_objective',levs,wpmean,wpstd,wpskew,wpkurt)
        print(' ')

    return


# RESEARCH RECORDKEEPING

# In[9]:


def print_parameters2(params):
    for key,value in params.items():
        print("{0}: {1}".format(key,value))
    return


# READ METADATA FOR USE IN WEIGHTED OPTIMIZATION

# In[10]:



def read_metadata(in_paramfile=None,uweights=None,sources_df=None):
    #read parameters from return_analyze program if input is assumed to be prediction returns from it
    #combine with sources_df stripped from return data file
    if not in_paramfile is None:
        try:
            with open(in_paramfile,'r') as infile:
                in_params=json.load(infile)
             
            #get tdiscount log to subtract from utility
            init_scenario_dict_df=pd.DataFrame(in_params.get('scenario_dict')).T
            scenario_dict_df=pd.DataFrame(init_scenario_dict_df[['interval','tdiscount']])

            # calculate time discount assuming interval data in months
            scenario_dict_df['log_tdiscount']=[-x[0]*np.log1p(x[1])/12.0 for x in scenario_dict_df.values]
            
            #get count of obs in each scenario without testing for valid keys
            counts=sources_df['dist_source'].value_counts()
            counts_df=pd.DataFrame([counts]).T
            counts_df.columns=['count']
            scenario_dict_df=scenario_dict_df.join(counts_df)
       
            #get desired scenario weights and translate to rows
            uweights_df=pd.DataFrame(uweights).T
            uweights_df.columns=['weight'] 
            
            #adjust weights to quasi-probabilities
            sumwt=np.sum(uweights_df['weight'].values)
            uweights_df['weight']= uweights_df['weight']/sumwt
            
            scenario_dict_df=scenario_dict_df.join(uweights_df)
            scenario_dict_df['row_weight']=scenario_dict_df['weight']/scenario_dict_df['count']

            intervals=scenario_dict_df['interval'].values #save interval information
            scenario_dict_df=scenario_dict_df[['log_tdiscount','row_weight']] 
            print(' ')
            print('INPUT SCENARIO METADATA')
            print(scenario_dict_df)
            
            #join with sources_df stripped from return data file
            scenario_dict_df['dist_source']=scenario_dict_df.index.values       
            sources_df=sources_df.join(scenario_dict_df.set_index('dist_source'),on='dist_source')

            # check to see if okay to do mean-variance comparison
            temp=[x for i,x in enumerate(intervals) if uweights_df['weight'].values[i] > 0 ]
            valid_mv=(max(temp)==min(temp))
            print(' ')
            print('Okay to compare with mean-variance is ',valid_mv)
            
            return(sources_df,valid_mv,init_scenario_dict_df,uweights_df)           
        except:
            print('Good param file not found.')
            raise


# MAIN PROGRAM

# In[11]:


def research_optimizer(params={}):    
    journalfile=params.get('journalfile')
    logfile=params.get('logfile')
    sample=params.get('sample')
    sourcefile=params.get('sourcefile')
    in_paramfile=params.get('in_paramfile')
    Llist=params.get('Llist')
    worst=params.get('worst')
    logging=params.get('logging')
    verbose=params.get('verbose')
    clusterfile=params.get('clusterfile')
    uweights=params.get('uweights')    
    
    #See import dependencies in first cell
    if logging:
        #record run description in journalfile
        orig_stdout = sys.stdout
        e=open(journalfile, 'a')
        sys.stdout=e
        print(' ')
        print_parameters2(params)
        e.close()
        
        #record results in logfile
        f = open(logfile, 'w')
        sys.stdout = f
    
        #record control parameters
        print_parameters2(params)
             
    #Read in Returns
    returns_df=load_source(sourcefile)
    
    #Extract scenario source
    print(' ')
    sources_df=None
    col0=returns_df.iloc[:,0]   #first column
    if isinstance(col0.iloc[0],str):
        sources_df=pd.DataFrame(col0)
        sources_df.columns=['dist_source']       
        returns_df=returns_df.drop([returns_df.columns.values[0]],axis=1)
    
    #read parameters from return_analyze program
    meta_df, valid_mv, init_scenario_dict_df, uweights_df=read_metadata(in_paramfile,uweights,sources_df)
    print('')
        
    if verbose and valid_mv:
        sdict=pd.concat([init_scenario_dict_df,uweights_df], axis=1)
        temp=[x[1] for x in sdict[['weight','interval']].values if x[0]>0.0]
        if len(temp)>0 and temp[0]==1:
            descript_df,corrs_df,count=rtn_summary(returns_df)
            print('INPUT RETURN MATRIX PARAMETERS')
            print(descript_df)
            
    # Do mean-variance and log leveraged surplus optimizations
    (wpreturns_df,mpreturns_df,learn_walloc,learn_malloc,learn_W_merit,learn_M_merit) = find_best_allocation(
        returns_df,worst,Llist,meta_df,valid_mv)
    if verbose:
        #describe portfolio return statistics for different leverages
        describe_portfolio_returns(wpreturns_df.values,
            mpreturns_df.values,learn_W_merit,learn_M_merit,Llist,verbose,valid_mv)

    print(' ')
    if logging:
        #close logfile and print it on terminal
        f.close()
        sys.stdout = orig_stdout

        h=open(logfile,'r')
        for line in h:
            if len(line)>0:          
                print(line[:-1])          
        h.close()
    print(' ')
    return 


# MAIN: SET PARAMETERS AND CALL RESILIENCE OPTIMIZER

# In[12]:


params=dict(
    journalfile='JOURNAL.txt',
    logfile='run001B.txt',
    sample='GITHUB EXAMPLE',
    sourcefile='prediction_01.csv',
    in_paramfile='params01.json',
    Llist=[1,2,4,8,16],
    worst=(-0.99),
    logging=True,
    verbose=True,
    uweights={'history':[0.4],'breakdown':[0.2],'6_month':[0.4]},
    )

#run main program
optimizer_output=None
try:   
    optimizer_output=research_optimizer(params)   
    print('DONE!')
except:
    print('Goodbye!!')
    print(' ')
    sys.exit()
    


# In[ ]:




