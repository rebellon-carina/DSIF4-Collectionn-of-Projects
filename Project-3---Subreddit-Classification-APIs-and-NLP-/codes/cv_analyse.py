from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class cv_analyse:
    '''
    handle processing for CountVectorizer output and analysis, default for CountVectorizer is stop_words= English, ngram is 1,2,3 and max_words=2000
    
    parameter to initialize: name of the feature, ngram list [1,2,3], and the max_features (maximum number of words)
    
    output the dataframe as word and count
    '''
    
    def __init__(self, X_feature, title="Untitled", list_ngram=[1,2,3], max_words=2000):
        self.X_feature = X_feature
        self.max_words = max_words
        self.ngram = list_ngram
        self.dict_bag_of_words = {}
        self.bag_of_words = {}
        self.title = title
        
        for ngram in list_ngram:
            cvec = CountVectorizer(ngram_range=(ngram,ngram), 
                               stop_words='english', max_features=self.max_words) 
            cvec.fit(self.X_feature)
            X_vec = cvec.transform(self.X_feature)
            self.bag_of_words[ngram] = pd.DataFrame(X_vec.toarray(), columns=cvec.get_feature_names())
            
            self.dict_bag_of_words[ngram] = pd.DataFrame(self.bag_of_words[ngram].sum().sort_values(ascending=False).reset_index().rename(
        columns={'index':'word', 0:'count'}))
        
        
    def get_top_N_words(self, ngram, topN, showChart=True):
        '''
        parameter is ngram (as integer, within the list sent during initialization)
        , and the top number words (top 10, 15,, etc)
        and showChart is boolean, True to show graph/chart, or False to return as dataframe
        '''
        if ngram in self.ngram:
            if showChart:
                return self.__top_N_chart(ngram,topN)
            else:
                return self.dict_bag_of_words[ngram].head(topN)
        else:
            raise InvalidNgram("Ngram is not in the initialization!")
    
    def __top_N_chart(self,ngram, topN):
        plt.title(f"Top {topN} Words for {self.title} (Ngram = {ngram} )", fontsize = 12)
        sns.barplot(data=self.dict_bag_of_words[ngram].head(topN), y="word", x="count" ) #, color='gray');
        pass

class InvalidNgram(Exception):
        pass
                     