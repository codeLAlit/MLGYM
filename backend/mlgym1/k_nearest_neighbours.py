import numpy as np
import pandas as pd

import warnings
from collections import Counter
import random

#Data is in form of Dictionary; Works for multiple test sets. 

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total classes!')
    vote_results = []
    for i in range(len(predict)):
        distances = []
        for group in data: #Individual classes of the dataset 
            for features in data[group]: #Individual entries in that particular class
                current_distance = np.linalg.norm(np.array(features) - np.array(predict[i]))
                distances.append([current_distance, group])

        distances = sorted(distances)
        votes = (i[1] for i in distances[:k])
        vote_results.append(Counter(votes).most_common(1)[0][0])

    return vote_results



def knn_run(df1, num_iter =5, k=3, test_size = 0.2):

    accuracies = []
    precisions_0 = []
    recalls_0 = []
    f1s_0 = []
    precisions_1 = []
    recalls_1 = []
    f1s_1 = []

    
    #Random Oversampling

    count_class_0, count_class_1 = df1.iloc[:,-1].value_counts()

    df_class_0 = df1[df1.iloc[:,-1] == 0]
    df_class_1 = df1[df1.iloc[:,-1] == 1]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df = pd.concat([df_class_0, df_class_1_over], axis=0)

    df = df.astype(float).values.tolist()

    for i in range(num_iter):

        random.shuffle(df)

        train_set = {0:[] , 1:[]} #Create a dictionary 
        test_set = {0:[], 1:[]}

        train_data = df[:-int(test_size*len(df))] #Get training data
        test_data = df[-int(test_size*len(df)):] #Get remaining as test data

        for i in train_data:
            train_set[i[-1]].append(i[:-1]) # Fill every training example into the relevant key of dictionary (0 or 1)
        for i in test_data:
            test_set[i[-1]].append(i[:-1])  # Fill every test example into the relevant key of dictionary (0 or 1)


        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0


        for group in test_set:
            vote_results = k_nearest_neighbors(train_set, test_set[group], k)
            for i in range(len(vote_results)):
                if vote_results[i] == group:
                    if group == 0:
                        true_negative += 1
                    else:
                        true_positive += 1
                else:
                    if group == 0:
                        false_positive += 1
                    else:
                        false_negative += 1

        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        
        precision_1 = true_positive / (true_positive + false_positive)
        recall_1 = true_positive / (true_positive + false_negative)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
        
        precision_0 = true_negative / (true_negative + false_negative)
        recall_0 = true_negative / (true_negative + false_positive)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)

        accuracies.append(accuracy)
        precisions_1.append(precision_1)
        recalls_1.append(recall_1)
        f1s_1.append(f1_1)
        precisions_0.append(precision_0)
        recalls_0.append(recall_0)
        f1s_0.append(f1_0)


    accuracy = sum(accuracies) / len(accuracies)
    #print("Accuracy : ", accuracy)
    
    precision_0 = sum(precisions_0) / len(precisions_0)
    #print("Precision for class 0 : ", precision_0)
    
    recall_0 = sum(recalls_0) / len(recalls_0)
    #print("Recall for  class 0 : ", recall_0)
    
    f1_0 = sum(f1s_0) / len(f1s_0)
    #print("F1 score for class 0 : ", f1_0)
    
    precision_1 = sum(precisions_1) / len(precisions_1)
    #print("Precision for class 1 : ", precision_1)
    
    recall_1 = sum(recalls_1) / len(recalls_1)
    #print("Recall for class 1 : ", recall_1)
    
    f1_1 = sum(f1s_1) / len(f1s_1)
    #print("F1 score for class 1 : ", f1_1)
    results = { "accuracy" : accuracy , "precision_0" : precision_0, "recall_0" : recall_0, "f1_0" : f1_0, "precision_1" : precision_1, "recall_1" :recall_1, "f1_1" : f1_1}
    return results 










    

    

    
