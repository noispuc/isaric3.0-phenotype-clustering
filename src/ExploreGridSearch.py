""" Class to explore the stepmix object list generated in the grid search function 
Code to explore the obj_list generated in the grid search
The class **optimize_stepMix** takes the outputs from gridSearch function and enable visualizations to explore results


# future improvement ideas:
merge this class with that grid search code 
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from stepmix.stepmix import StepMix

class optimize_stepMix():
    def __init__(self, obj_list,stats,
                 data,
                 predictors,outcome):
        X=data.loc[:,predictors]
        Y=data.loc[:,outcome]
        k_list=[obj.n_components for obj in obj_list]
        stepMixObjs=dict(zip(k_list,obj_list))
        self.modelObjs=stepMixObjs
        self.k_list=k_list
        self.stats=stats
        self.X=X
        self.Y=Y
        self.data0=data
        self.predictors=predictors
        
    def summary(self,k,X=None,Y=None,covariates=None,resultsDir=None,return_predValue=False):
        if X is None:
            X=self.X
        if Y is None:
            Y=self.Y
        if resultsDir is not None:
            if not os.path.exists(resultsDir):
                os.makedirs(resultsDir)
            if not os.path.exists(f'{resultsDir}/ncomp_{k}'):
                os.makedirs(f'{resultsDir}/ncomp_{k}')
        # print statistics
        print(f'======================= LCA with k = {k}=========================')
        model_k=self.modelObjs[k]
        model_k.report(X)
        #stats=self.stats
        #print(stats[stats.k==k].T)
        # Conditional probabilities
        #print('------------------------ Conditional probabillities ------------------------------')
        mmdf=model_k.get_mm_df()
        mmdf2=mmdf.reset_index(level=['model_name','param'],drop=True)
        mmdf2.reset_index(drop=False,inplace=True)
        mmdf2=mmdf2.set_index('variable')
        #print(mmdf2)
        mmdf=mmdf.reset_index()

        # line plot ======================
        fig = plt.figure(figsize=(20,8))
        ax=mmdf.drop(['model_name','param','variable'],axis=1).plot(kind='line',figsize=(15,5))
        #ax.set_xticklabels(varList)
        plt.xticks(ticks=range(len(self.predictors)), labels=mmdf.variable, rotation=45)
        if resultsDir is not None:
            #plt.savefig(f'{resultsDir}/ncomp_{k}/allvar_line.png', bbox_inches='tight')
            mmdf.to_csv(f'{resultsDir}/ncomp_{k}/conditionalProb.csv')
        plt.show()
        # radar plot =========
        mmdf=mmdf.reset_index(drop=False)
        ticks=[x for x in mmdf.columns[1:].values]
        categories=mmdf.variable
        N=len(categories)
        from math import pi
        
        for i, v in enumerate(range(k)):
            # But we need to repeat the first value to close the circular graph:
            values=mmdf[v].tolist()
            values += values[:1]
            values
        
            # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            fig, ax = plt.subplots(1,figsize=(5,5), subplot_kw={'projection': 'polar'})
            
            # Draw one axe per variable + add labels
            plt.xticks(angles[:-1], categories, color='blue', size=12)
        
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.1,0.5,1], ["0.1","0.5","1"], color="grey", size=7)
            plt.ylim(0,1)
        
            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid')
        
            # Fill area
            ax.fill(angles, values, 'darkblue', alpha=0.1)
            ax.set_title(f'class {v}')
        
            # Show the graph
            if resultsDir is not None:
                plt.savefig(f'{resultsDir}/ncomp_{k}/{v}_radar.png', bbox_inches='tight')
            #plt.show()

        # predictions
        print('------------------------ predictions ------------------------------')
        predicted=model_k.predict(X)

        ct=pd.Series(predicted).value_counts()
        count_df=pd.DataFrame({'cl': [f'cl_{x}' for x in ct.index], 'n':ct, 'perc':round(ct/sum(ct)*100,2)})
        lbl=[f'{p}%\n N={x}' for p,x in zip(count_df.perc,count_df.n)]
       
        fig, ax = plt.subplots(figsize=(8,5))
                    
        bars=ax.barh(count_df.cl, count_df.perc)
        #plt.bar_label(lbl, padding=5)
        ax.bar_label(bars, labels=lbl,
                     padding=1, color='b', fontsize=8)
        plt.ylabel('Classes')
        plt.xlabel('Percentages')
        plt.xlim(0,max(count_df.perc+10))
        plt.title(f'Frequencies')
        if resultsDir is not None:
            plt.savefig(f'{resultsDir}/ncomp_{k}/Frequecies.png', bbox_inches='tight')
        plt.show()
        
        print(pd.crosstab(Y, predicted,normalize='columns'))

        if return_predValue==True:
            return predicted
        
    def decide(self, k=None):
        if k is None:
            return 'k must be informed'
        else:
            #TOOD Issue alert if overwriting 
            self.k=k
            model_k=self.modelObjs[k]
            X=self.X
            self.otpimized_object=model_k
            data=self.data0
            predicted=model_k.predict(X)
            data['predicted']=predicted
            self.predicted_Y=predicted
            data['predicted'] = data['predicted'].astype('category').cat.codes
            self.data=data
            print(f'decision on k={k} stored')


