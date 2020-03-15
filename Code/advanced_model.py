
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:55:05 2019

@author: Hitesh
"""


"Explorartory data analysis on the dataset"
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

df = pd.read_excel('Oct-18_Call Dump.xlsx')

import re
#import nltk
from my_normalisation import normalize_corpus
from feature_extractors import bow_extractor
from sklearn.model_selection import train_test_split


import pickle


"taking columns that are needed"
#df_new =pd.DataFrame(df_small,columns=['Summary', 'Description', 'Category','Sub-Category','labels'])

#############################################################################################################################################
"First trial model keeping summary as X and Sub-Category as Y"
"cleaning summary"

summary_list=df['Summary'].tolist()
 

##try_list = summary_list[1:100]
labels = df['Sub-Category'].tolist()

"checking for null documents"
def remove_empty_files(corpus,labels):
    new_corp = []
    new_files =[]
    for doc, labels in zip(corpus,labels):
        if doc.strip():
            new_corp.append(doc)
            new_files.append(labels)
    return new_corp, new_files


########
labels = ['zz'if x is np.nan else x for x in labels]

    

##trying for split size 20%
def prepare_datasets(corpus, labels, test_data_proportion=0.2):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, 
                                                        test_size=0.2, random_state=42)
    return train_X, test_X, train_Y, test_Y




#######################################################################################################################################################
##TRIAL FOR GOOGLR WORD VEC MODEL    


    




"refer to normalise corpus in my_normalisation.py file"
#####google word_vec model implemented in genism
#import gensim
#import nltk
#tokenized_train = [nltk.word_tokenize(text) for text in norm_train_summary]
#tokenized_test = [nltk.word_tokenize(text) for text in norm_test_summary]
#model = gensim.models.Word2Vec(tokenized_train,size=500,window=100,min_count=30,sample=1e-3) 

#import numpy as np

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector,model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)        
        
    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary,num_features) for tokenized_sentence in corpus]
    return np.array(features)
    
####averaged word features
#avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,model=model,num_features=500)
#avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,model=model,num_features=500)


#svm_avgwv_predictions = train_predict_evaluate_model(classifier=sgd,train_features=avg_wv_train_features, train_labels=train_labels,test_features=avg_wv_test_features, test_labels=test_labels)
#mnb_avgwv_predictions = train_predict_evaluate_model(classifier=mnb,train_features=avg_wv_train_features, train_labels=train_labels,test_features=avg_wv_test_features, test_labels=test_labels)






###TRIAL FOR GOOGLR WORD VEC MODEL WE got 77% accuracy from that
#########################################################################################################################################################3

def my_metrics(pred_labels,actual_labels):
    print('accuracy:',np.round(metrics.accuracy_score(pred_labels,actual_labels),2))
    print('accuracy:',np.round(metrics.accuracy_score(pred_labels,actual_labels),2))
    
def train_predict_evaluate_model(classifier,train_features,train_labels,test_features,test_labels):
    classifier.fit(train_features,train_labels)
    predictions = classifier.predict(test_features)
    my_metrics(actual_labels=test_labels,pred_labels=predictions)
    return predictions

"making instances of classfiers"
mnb=MultinomialNB()
sgd=SGDClassifier(loss='hinge', n_iter=100)


##bow model gave me 80% accuracy with mnb classifier
#bow_predictions = train_predict_evaluate_model(classifier=mnb,train_features=bow_train_features, train_labels=train_labels,test_features=bow_test_features, test_labels=test_labels)

#tfidf model ,mnb 78%
#tfid_predictions = train_predict_evaluate_model(classifier=mnb,train_features=tfidf_train_features, train_labels=train_labels,test_features=tfidf_test_features, test_labels=test_labels)

##bow model gave me 84% accuracy with support vector machine classifier

#tfidf model 83%
#tfid_predictions_sgd = train_predict_evaluate_model(classifier=sgd,train_features=tfidf_train_features, train_labels=train_labels,test_features=tfidf_test_features, test_labels=test_labels)

#########################################################################################################################################

"confusion matrix to visualise the predictions"



#cm = metrics.confusion_matrix(test_labels,my_pred)


def get_unique_list(l_old):
    l_new = []
    for item in l_old:
        if item not in l_new: l_new.append(item)
    return l_new

#unique_test_list = sorted(get_unique_list(test_labels))
#unique_train_list = sorted(get_unique_list(train_labels))


############################################################################################################################################
"fixing to boost accuracy"
"desktop and laptops similarity"

def compare_summary(test_corpus,test_labels,pred_labels):
    n= len(test_labels)
    num =0
    x=[]
    for i in range(n):
        if num<=100:         
            if test_labels[i] =='Desktop' and pred_labels[i] =='Laptops':
                x.append(test_corpus[i])
                num = num + 1
        else:
            break
    return x 
        
#x_diff = compare_summary(test_summary,test_labels,my_pred)
        


############################################################################################################################################3
"Building use case for Printer configs"
def sample_mails(summary,category):
    n=0
    x=[]
    while n<=4:     
        for i in range(4000):
            if category[i]=='Printer':     
                x.append(summary[i])
                n=n+1
    return x    

#y=sample_mails(test_summary,test_labels)    
#####################################################################################################################################################
 
"Entity extraction from printer type mails"
"printer number is in capitals"

"need a new column for asset id" 
"try one regex pattern" "working"

def get_entity(summary,labels):
    if labels=="Printer":
        printer_num = re.findall(r'[A-Z0-9]{4,10}', summary)
        return printer_num

with open("new_email.txt", "r") as f:
    data = f.readlines()




############################################################################################################################3
def entity_extraction(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    data_bow_features = bow_vectorizer.transform(data)
    pred_label = sgd.predict(data_bow_features)
    entity= get_entity(data[0],labels=pred_label)
    return entity


#from sklearn.externals import joblib
#joblib.dump(sgd, 'model.pkl')




if __name__ == '__main__':
    summary, labels = remove_empty_files(summary_list,labels)
    train_summary,test_summary,train_labels,test_labels = prepare_datasets(summary,labels)
    norm_train_summary =normalize_corpus(train_summary,lemmatize=False,tokenize=False)
    norm_test_summary =normalize_corpus(test_summary,lemmatize=False,tokenize=False)
    
    ###bag of words model
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_summary)  
    bow_test_features = bow_vectorizer.transform(norm_test_summary)
    pickle.dump(bow_vectorizer, open("vector.pickel", "wb"))
    
    #tfidf model
    #tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_summary)  
    #tfidf_test_features = tfidf_vectorizer.transform(norm_test_summary)
    
    bow_predictions_sgd = train_predict_evaluate_model(classifier=sgd,train_features=bow_train_features, train_labels=train_labels,test_features=bow_test_features, test_labels=test_labels)
    my_pred = bow_predictions_sgd.tolist()
    zy = entity_extraction(filename="new_email.txt")
    
    















    































