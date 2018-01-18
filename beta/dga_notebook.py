import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import sklearn.feature_extraction
import operator
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from collections import Counter
from itertools import groupby

"""
Reading in Input
"""

def get_dga_data_one():
    dga_dataframe = pd.read_csv('dga.csv', index_col = False)
    dga_dataframe = dga_dataframe[dga_dataframe['class'] == 'dga']
    dga_dataframe.drop(['host', 'subclass'], axis = 1, inplace = True)
    return dga_dataframe;

def get_dga_data_two():
    dga_file = open("dga_input_two.txt", "r").readlines()
    dga_file = [line.split()[1] for line in dga_file]
    dga_dataframe = pd.DataFrame(data = {'domain' : dga_file})
    dga_dataframe['class'] = 'dga'
    dga_dataframe['domain'] = dga_dataframe['domain'].apply(lambda x : x.split('.')[0].lower())
    return dga_dataframe

def get_legit_data():
    legit_file = open("dns_top10000_domains_20170308.txt", "r").readlines()
    legit_dataframe = pd.DataFrame(data = {'domain' : legit_file})
    legit_dataframe['class'] = 'legit'
    legit_dataframe['domain'] = legit_dataframe['domain'].apply(lambda x : x.split('.')[0].lower())
    return legit_dataframe

'''
Reading in input as a dataframe (df). You can choose to use get_dga_data_one which contains data of only 2 - 3 dgas and is much easier to classify. This produces an accuracy of ~99% for the detection algorithm. Or you can use get_dga_data_two which has 11+ dgas in the input and is a little harder to classify. The accuracy detection rate is about 96% for this input set. 
'''
def read_input():
    #dga_df = get_dga_data_one()
    dga_df = get_dga_data_two()
    legit_df = get_legit_data()
    legit_df.drop_duplicates(subset = 'domain', inplace = True)
    legit_df = legit_df.sample(frac = 1).reset_index(drop = True)
    legit_df.dropna()
    dga_df.drop_duplicates(subset = 'domain', inplace = True)
    dga_df = dga_df.sample(n = 10000).reset_index(drop = True)
    dga_df.dropna()
    #print dga_df.describe()
    df = pd.concat([legit_df, dga_df])
    df = df.sample(frac = 1.0).reset_index(drop = True)
    return df

"""
Extracting features
"""

def ngram_count(domain, dns_counts, dns_vc):
    total_match = dns_counts * dns_vc.transform([domain]).T
    print '%.2f total matches' %(total_match)

def calc_entropy(s):
    p, length = Counter(s), float(len(s))
    return -sum([count / length * math.log(count / length, 2) for count in p.values()])

def calc_common_top_level_ngram():
    dns_file = open("dns_top10000_domains_20170308.txt", "r").readlines()
    dns_dataframe = pd.DataFrame(data = {'domain' : dns_file})
    dns_dataframe['domain'] = dns_dataframe['domain'].apply(lambda x : x.split('.')[0].lower())
    dns_dataframe.drop_duplicates(subset = 'domain', inplace = True)
    dns_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer = 'char', ngram_range = (2, 3), min_df = 3, max_df = 1.0)
    document_matrix = dns_vc.fit_transform(dns_dataframe['domain'])
    dns_counts = np.log(document_matrix.sum(axis = 0).getA1())
    ngrams_list = dns_vc.get_feature_names()
    sorted_ngrams = sorted(zip(ngrams_list, dns_counts), key = operator.itemgetter(1), reverse = True)
    ngram_map = dict(zip(ngrams_list, dns_counts))
    #for ngram, count in sorted_ngrams:
    #    print '{} {}'.format(ngram, count)
    return sorted_ngrams, dns_counts, dns_vc, ngrams_list, ngram_map
    
def calc_ngram(word, ngram_length, ngram_map):
    if len(word) < ngram_length:
        return 0.0
    total = 0.0;
    for i in range(0, len(word) - ngram_length + 1):
        if word[i : i + ngram_length] in ngram_map:
            total += ngram_map[word[i : i + ngram_length]]
    return total / float(len(word) - ngram_length + 1)
                       
def calc_vowels(word):#how many a,e,i,o,u
    vowels = list('aeiou')
    return float(sum(vowels.count(i) for i in word))

def calc_digits(word):#how many digits
    digits = list('0123456789')
    return float(sum(digits.count(i) for i in word.lower()))

def calc_repeat_letter(word):#how many repeated letter
    count = Counter(i for i in word.lower() if i.isalpha()).most_common()
    cnt = 0.0
    for letter, ct in count:
        if ct > 1:
            cnt += 1
    return float(cnt)

def calc_consecutive_digits(word):#how many consecutive digit
    digit_map = [ int(i.isdigit()) for i in word ]
    consecutive=[ (k, len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i, j in consecutive if j > 1 and i == 1)
    return float(count_consecutive)

def calc_consecutive_consonant(word):#how many consecutive consonant
    consonant = set(['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w','x', 'y', 'z'])
    digit_map = [int(i in consonant) for i in word]
    consecutive=[(k, len(list(g))) for k, g in groupby(digit_map)]
    count_consecutive = sum(j for i,j in consecutive if j > 1 and i == 1)
    return float(count_consecutive)
       
'''
Feature extractor
calculates the entropy (shannon), bigram (average), trigram (average), length, # of vowels, # of digits, # of repeated letters, # of consecutive digits, and # of consecutive consonants
'''

def feature_extractor(df):
    df['entropy'] = df['domain'].apply(lambda x : calc_entropy(x))
    sorted_ngrams, dns_counts, dns_vc, ngrams_list, ngram_map = calc_common_top_level_ngram()
    df['bigram'] = df['domain'].apply(lambda x : calc_ngram(x, 2, ngram_map))
    df['trigram'] = df['domain'].apply(lambda x : calc_ngram(x, 3, ngram_map))
    df['length'] = df['domain'].apply(lambda x : float(len(x)))
    df['vowels'] = df['domain'].apply(lambda x : calc_vowels(x) / len(x))
    df['digits'] = df['domain'].apply(lambda x : calc_digits(x) / len(x))
    df['repeat_letter'] = df['domain'].apply(lambda x : calc_repeat_letter(x) / len(x))
    df['consecutive_digit'] = df['domain'].apply(lambda x : calc_consecutive_digits(x) / len(x))
    df['consecutive_consonant'] = df['domain'].apply(lambda x : calc_consecutive_consonant(x) / len(x))
 
    '''this previous calculation on the first iteration of ngram is inaccurate and is similar to length because if the length is high then the ngram will be high, instead you should take the average of the bigrams
    df['ngram'] = dns_counts * dns_vc.transform(df['domain']).T'''
    
    return df
     
"""
Graphing features
"""

def graph_entropy_feature_boxplot(df):
    plt.figure()
    df.groupby('class').boxplot()
    plt.show()
    
def scatter_plot_length_vs_feature(dga, legit, feature):
    fig, ax1 = plt.subplots(1, 1, sharey = True)
    ax1.scatter(legit['length'], legit[feature], s = 120, c = '#aaaaff', label = 'legit', alpha = 0.2, edgecolor = 'black')
    ax1.scatter(dga['length'], dga[feature], s = 120, c = 'red', label = 'dga', alpha = 0.2, edgecolor = 'black')
    ax1.legend()
    ax1.set_title('length versus ' + feature)
    ax1.set_xlabel('length')
    ax1.set_ylabel(feature)
    plt.show() 
    
def hist_plot_feature(dga, legit, feature):
    plt.hist([dga[feature], legit[feature]], bins = 100, color = ['b','r'], label = ['dga', 'legit'], alpha = 0.5)
    plt.title('dga ' + feature + ' vs legit ' + feature)                          
    plt.xlabel('feature')
    plt.ylabel('count')
    plt.legend()                                                      
    plt.show()
    
#graphs all features   
#uncomment which plots and graphs you want to see
def graph_features(df):
    dga, legit = df[df['class'] == 'dga'], df[df['class'] == 'legit']
    #graph_entropy_feature_boxplot(df)
    #hist_plot_feature(dga, legit, 'bigram')
    #hist_plot_feature(dga, legit, 'trigram')
    #hist_plot_feature(dga, legit, 'length')
    #hist_plot_feature(dga, legit, 'vowels')
    #hist_plot_feature(dga, legit, 'repeat_letter')
    #hist_plot_feature(dga, legit, 'consecutive_digit')
    #hist_plot_feature(dga, legit, 'consecutive_consonant')
    #scatter_plot_length_vs_feature(dga, legit, 'consecutive_digit')
    #scatter_plot_length_vs_feature(dga, legit, 'consecutive_consonant')
    #scatter_plot_length_vs_feature(dga, legit, 'bigram')
    #scatter_plot_length_vs_feature(dga, legit, 'trigram')
    #scatter_plot_length_vs_feature(dga, legit, 'length')
    #scatter_plot_length_vs_feature(dga, legit, 'vowels')
    #scatter_plot_length_vs_feature(dga, legit, 'digits')
    #scatter_plot_length_vs_feature(dga, legit, 'repeat_letter')
    
"""
Modeling and Testing Features
"""

def print_score(accuracy_scores, precision_scores):
    print 'Mean Accuracy Score: %f' % (accuracy_scores.mean())
    print 'All Cross Validation Accuracy Scores {}'.format(accuracy_scores)
    print 'Mean Precision Score: %f' % (precision_scores.mean())
    print 'All Cross Validation Precision Scores {}'.format(precision_scores)
    
#10 fold cross validation scoring
def cross_validation_score(X, y, clfs, features):
    for clf in clfs:
        accuracy_scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv = 10, n_jobs = -1, scoring = 'accuracy')
        precision_scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv = 10, n_jobs = -1, scoring = 'f1')
        if isinstance(clf, sklearn.ensemble.RandomForestClassifier):
            print 'Random Forest Classifier Results'
            print_score(accuracy_scores, precision_scores)
        elif isinstance(clf, sklearn.svm.SVC):
            print 'SVM Classifier Results'
            print_score(accuracy_scores, precision_scores)

#shows what features the svm model thinks are important    
def graph_svm_feature_importances(coef, names):
    plt.figure()
    imp, names = zip(*sorted(zip(coef, names)))
    #bar works like x range and height = y range, barh works in the EXACT same way
    plt.bar(range(len(names)), height = imp, align = 'center')
    #label x not y ticks with the range -> names mapping
    plt.xticks(range(len(names)), names)
    plt.show()   

#shows what features the random forest model thinks are important
def graph_random_forest_feature_importances(coef, names):
    plt.figure()
    imp, names = zip(*sorted(zip(coef, names)))
    plt.bar(range(len(names)), height = imp, align = 'center')
    plt.xticks(range(len(names)), names)
    plt.show()


'''   
Confusion Matrix of true negatives, true positives, false negatives, and false positives. Good for analyzing results
'''
def plot_cm(cm, labels):
    tn, fp, fn, tp = cm.ravel()
    print 'Percentage scores'
    print '%s/%s: %f (%d/%d)' %(labels[0], labels[0], (100.0 * tn) / (tn + fp), tn, tn + fp)
    print '%s/%s: %f (%d/%d)' %(labels[0], labels[1], (100.0 * fp) / (tn + fp), fp, tn + fp)
    print '%s/%s: %f (%d/%d)' %(labels[1], labels[0], (100.0 * fn) / (tp + fn), fn, tp + fn)
    print '%s/%s: %f (%d/%d)' %(labels[1], labels[1], (100.0 * tp) / (tp + fn), tp, tp + fn)
    
    cumsum = cm.sum(axis = 1)
    cumsum = np.array(cumsum, dtype = 'float')
    percentages = (100.0 * cm) / cumsum
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(percentages, cmap = 'coolwarm')
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()    
    
    
'''
Calculates an 80 - 20 training vs testing model with the classifiers (clfs)
X is features, y is test result. features is the name of features trained
single score accuracy result is printed and feature importances and confusion matrix are graphed.
'''
def single_split_score(X, y, clfs, features):
    for clf in clfs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        single_score = accuracy_score(y_test, y_pred)
        
        if isinstance(clf, sklearn.ensemble.RandomForestClassifier):
            print 'Random Forest Single Split Score: %f' %(single_score)
            print 'Random Forest Feature importances!'
            for name, importance in zip(features, clf.feature_importances_):
                print 'Feature %s has importance: %f' % (name, importance)
            graph_random_forest_feature_importances(clf.feature_importances_, features)
        elif isinstance(clf, sklearn.svm.SVC):
            print 'SVM Single Split Score: %f' %(single_score)
            print 'Feature importances!'
            for name, importance in zip(features, clf.coef_.ravel()):
                print 'Feature %s has importance: %f' % (name, importance)
            graph_svm_feature_importances(clf.coef_.ravel(), features)
        
        cm = confusion_matrix(y_test, y_pred)
        plot_cm(cm, ['dga', 'legit']) 


'''
SVM and RandomForest models are tested and graphed. They give accuracies of ~96% and 99% on the second and first dga set respectively (get_dga_data_two(), get_dga_data_one())
Parameters were not tuned using GridSearchCV and so accuracy can be slightly improved by testing different parameters. SVC was implemented with linear kernel for speed but a rbf kernel should be more accurate
'''
def test_and_graph_model(df):
    svm = SVC(kernel = 'linear', C = 1, gamma = 0.01, probability = True, random_state = 0)
    rf = RandomForestClassifier(min_samples_split = 3, n_estimators = 5, max_depth = 5, min_samples_leaf = 1)
    
    features = ['length', 'entropy', 'bigram', 'trigram', 'vowels', 'digits', 'repeat_letter', 'consecutive_digit', 'consecutive_consonant']
    X = df.as_matrix(features)
    y = df['class'].apply(lambda x : 0.0 if x == 'dga' else 1.0)
    y = np.array(y.tolist())
    clfs = [rf, svm]
    
    single_split_score(X, y, clfs, features)
    cross_validation_score(X, y, clfs, features)
     
        

if __name__ == "__main__":
    df = read_input()
    df = feature_extractor(df)
    graph_features(df)
    test_and_graph_model(df)
    
    
    