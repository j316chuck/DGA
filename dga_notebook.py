import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import sklearn.feature_extraction
import operator
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from collections import Counter

"""
data and feature extraction phase
load data
extract length feature
extract entropy feature
extract ngram feature
"""

def entropy(s):
    p, length = Counter(s), float(len(s))
    return -sum([count / length * math.log(count / length, 2) for count in p.values()])

def ngram_count(domain):
    total_match = dns_counts * dns_vc.transform([domain]).T
    print '%.2f total matches' %(total_match)

#reading in dga
dga_dataframe = pd.read_csv('dga.csv', index_col = False)
dga_dataframe = dga_dataframe[dga_dataframe['class'] == 'dga']
dga_dataframe.drop(['host', 'subclass'], axis = 1, inplace = True)

#reading in legit 
legit_file = open("dns_top10000_domains_20170308.txt", "r").readlines()
legit_dataframe = pd.DataFrame(data = {'domain' : legit_file})
legit_dataframe['class'] = 'legit'
legit_dataframe['domain'] = legit_dataframe['domain'].apply(lambda x : x.split('.')[0].lower())


#cleaning data and shuffling
legit_dataframe.drop_duplicates(subset = 'domain', inplace = True)
legit_dataframe = legit_dataframe.sample(frac = 1).reset_index(drop = True)
dga_dataframe = dga_dataframe.sample(frac = 1).reset_index(drop = True)

#df dataset is comprised of 8000 legit samples, 2000 illegit samples
#80 20 split between dataset and test set - 2000 for hold out
#may do stratified random sampling next time to get better data
legit_dataframe = legit_dataframe[:8000]
dga_dataframe = dga_dataframe[:2000]
df = pd.concat([legit_dataframe, dga_dataframe])
df = df.sample(frac = 1.0).reset_index(drop = True) 
hold_df = df[:2000].reset_index(drop = True)
hold_df.dropna()
train_df = df[2000:].reset_index(drop = True)
train_df.dropna()

#adding features length, entropy, n-gram vectorization
#jitter makes it easier to see the distributions
#clearly by the strip plot, dga's have longer base names
train_df['length'] = train_df.apply(lambda row: len(row.domain), axis = 1)
hold_df['length'] = hold_df.apply(lambda row: len(row.domain), axis = 1)
sns.stripplot(x = "class", y = "length", data = train_df, jitter = True)
plt.show()
sns.violinplot(x = "class", y = "length", data = train_df)
plt.show() 
sns.barplot(x = "class", y = "length", data = hold_df)
plt.show() 


train_df['entropy'] = train_df['domain'].apply(lambda x : entropy(x))
hold_df['entropy'] = hold_df['domain'].apply(lambda x : entropy(x))
plt.figure()
train_df.groupby('class').boxplot()
plt.show()
plt.figure()
train_df.groupby('class')['entropy'].plot(legend = True)
plt.show()

#computing 3, 4, and 5 gram
#using top 10000 to get a vectorizer which can fit to a set of data, getting the 3, 4, and 5 gram
#and then transform a data into a set into a document-term matrix consisting of ones and zeros
dns_file = open("dns_top10000_domains_20170308.txt", "r").readlines()
dns_dataframe = pd.DataFrame(data = {'domain' : legit_file})
dns_dataframe['domain'] = dns_dataframe['domain'].apply(lambda x : x.split('.')[0].lower())
dns_dataframe.drop_duplicates(subset = 'domain', inplace = True)
dns_vc = sklearn.feature_extraction.text.CountVectorizer(analyzer = 'char', ngram_range = (3, 5), min_df = 3, max_df = 1.0)
document_matrix = dns_vc.fit_transform(dns_dataframe['domain'])
dns_counts = np.log(document_matrix.sum(axis = 0).getA1())
ngrams_list = dns_vc.get_feature_names()
sorted_ngrams = sorted(zip(ngrams_list, dns_counts), key = operator.itemgetter(1), reverse = True)
for ngram, count in sorted_ngrams:
    print '{} {}'.format(ngram, count)

    
ngram_count('google')
ngram_count('baidu')
ngram_count('weibo')
ngram_count('beyonce')
ngram_count('bey666on4ce')

train_df['ngram'] = dns_counts * dns_vc.transform(train_df['domain']).T
hold_df['ngram'] = dns_counts * dns_vc.transform(hold_df['domain']).T
train_df.sort_values(by = 'ngram', ascending = True).head(100)
train_df.describe()

conditional = train_df['class'] == 'dga'
dga = train_df[conditional]
legit = train_df[~conditional]


def scatter_plot_features():
    fig, ax1 = plt.subplots(1, 1, sharey = True)
    #plt.scatter(legit['entropy'], legit['alexa_grams'],  s=120, c='#aaaaff', label='Alexa', alpha=.2)
    ax1.scatter(legit['length'], legit['ngram'], s = 120, c = '#aaaaff', label = 'legit', alpha = 0.2, edgecolor = 'black')
    ax1.scatter(dga['length'], dga['ngram'], s = 120, c = 'red', label = 'dga', alpha = 0.2, edgecolor = 'black')
    ax1.legend()
    ax1.set_title('length versus ngram')
    ax1.set_xlabel('length')
    ax1.set_ylabel('ngram')
    plt.show()    
scatter_plot_features()

def line_plot_features():
    plt.figure()
    plt.plot(legit['length'], legit['ngram'], color = 'green', label = 'ngram')
    plt.plot(dga['length'], dga['ngram'], color = 'orange', label = 'dga')
    plt.legend()
    plt.title('length versus ngram')
    plt.xlabel('length')
    plt.ylabel('ngram')
    plt.show()    
line_plot_features()

def box_plot_features():
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
    ax1.boxplot(list(legit['ngram']))
    ax1.set_title('legit')
    ax2.boxplot(list(dga['ngram']))
    ax2.set_title('dga')
    plt.show()
box_plot_features()

def hist_plot_features():
    fig, ax = plt.subplots(1, 1)
    legit = train_df[train_df['class'] == 'legit']
    ax = legit['ngram'].hist(bins = 40)
    ax.figure.suptitle('Histogram of the NGram Score versus Domain')
    plt.xlabel('N gram score')
    plt.ylabel('Number of Domains')
    plt.show()    
hist_plot_features()    

"""
testing and modelling phase
vanilla model testing
cross validation scoring
train_test split scoring
confusion matrix plotting
feature importance plotting
testing three algorithms: naive bayes, random forest, and svm
"""

def test_clf(clf, X, y):
    #cross validation scoring
    accuracy_scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv = 5, n_jobs = -1, scoring = 'accuracy')
    print 'Mean Accuracy Score: %f' % (accuracy_scores.mean())
    print 'All Cross Validation Accuracy Scores {}'.format(accuracy_scores)
    
    precision_scores = sklearn.cross_validation.cross_val_score(clf, X, y, cv = 5, n_jobs = -1, scoring = 'f1')
    print 'Mean Precision Score: %f' % (precision_scores.mean())
    print 'All Cross Validation Precision Scores {}'.format(precision_scores)
    
    
    #single split scoring
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    single_score = accuracy_score(y_test, y_pred)
    print 'Single Split Score: %f' %(single_score)
    
    features = ['length', 'entropy', 'ngram']
    #feature importances
    if isinstance(clf, sklearn.ensemble.RandomForestClassifier):
        print 'Feature importances!'
        for name, importance in zip(features, clf.feature_importances_):
            print 'Feature %s has importance: %f' % (name, importance)
        random_forest_feature_importances(clf.feature_importances_, features)
    elif isinstance(clf, sklearn.svm.SVC):
        print 'Feature importances!'
        for name, importance in zip(features, clf.coef_.ravel()):
            print 'Feature %s has importance: %f' % (name, importance)
        svm_feature_importances(clf.coef_.ravel(), features)
        
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, ['dga', 'legit'])
    
    
def svm_feature_importances(coef, names):
    plt.figure()
    imp, names = zip(*sorted(zip(coef, names)))
    #bar works like x range and height = y range, barh works in the EXACT same way
    plt.bar(range(len(names)), height = imp, align = 'center')
    #label x not y ticks with the range -> names mapping
    plt.xticks(range(len(names)), names)
    plt.show()   
    
def random_forest_feature_importances(coef, names):
    plt.figure()
    imp, names = zip(*sorted(zip(coef, names)))
    plt.bar(range(len(names)), height = imp, align = 'center')
    plt.xticks(range(len(names)), names)
    plt.show()
    

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
    #ax.grid(b = False)
    cax = ax.matshow(percentages, cmap = 'coolwarm')
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    
X = train_df.as_matrix(['length', 'entropy', 'ngram'])
y = train_df['class'].apply(lambda x : 0.0 if x == 'dga' else 1.0)
y = np.array(y.tolist())

print 'Random Forest Classifier Results'
clf = RandomForestClassifier()
test_clf(clf, X, y)

print 'Multinomial Naive Bayes Results'
clf = MultinomialNB()
test_clf(clf, X, y)

print 'SVM Results'
clf = SVC(kernel = 'linear')
test_clf(clf, X, y)


def grid_search_eval(X, y, parameters, classifier, classifier_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = GridSearchCV(classifier, parameters, cv = 5, scoring = 'accuracy', n_jobs = -1)
    clf.fit(X_train, y_train)
    print 'Best parameters found in {} Grid Search'.format(classifier_name)
    print clf.best_params_
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print 'Score is %.3f (+/-%.3f) for parameters: %r' % (mean, std * 2, params)
        
    y_pred = clf.predict(X_test)
    single_score = accuracy_score(y_test, y_pred)
    print '%s Grid Search Score: %f' %(classifier_name, single_score)
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, ['dga', 'legit'])
    return clf
    
random_forest_parameters = {
    "n_estimators" : [5, 20, 50, 100],
    "max_depth" : [2, 5, 10],
    "min_samples_split" : [3, 10],
    "min_samples_leaf" : [1, 3, 10]
}

svm_parameters = {
    "kernel" : ['rbf', 'linear', 'poly'],
    "C" : [0.1, 1, 10, 100],
    "gamma" : [1e-4, 1e-3, 1e-2, 1e-1] 
}

random_forest_clf = grid_search_eval(X, y, random_forest_parameters, RandomForestClassifier(), 'Random Forest')
svm_clf = grid_search_eval(X, y, svm_parameters, SVC(), 'SVM')
    
#test one on hold out data
def hold_out_test(classifiers, classifier_names):
    hold_X = hold_df.as_matrix(['length', 'entropy', 'ngram'])
    hold_y = hold_df['class'].apply(lambda x : 0.0 if x == 'dga' else 1.0)
    hold_y = np.array(hold_y.tolist())
    
    scores = []
    for classifier, name in zip(classifiers, classifier_names):
        y_pred = classifier.predict(hold_X)
        score = accuracy_score(hold_y, y_pred)
        print '%s classifier score: %.10f' % (name, score)
        scores.append((score, name, classifier))
    scores.sort(key = lambda x : x[0], reverse = True)
    print 'best classifier {} with score {}'.format(scores[0][1], scores[0][0])
    cm = confusion_matrix(hold_y, scores[0][2].predict(hold_X))
    plot_cm(cm, ['dga', 'legit'])
    return scores[0][2]

svm_vanilla = SVC()
svm_vanilla.fit(X, y)
random_forest_vanilla = RandomForestClassifier()
random_forest_vanilla.fit(X, y)
random_forest_clf = RandomForestClassifier(min_samples_split = 3, n_estimators = 5, max_depth = 5, min_samples_leaf = 1)
random_forest_clf.fit(X, y)
svm_clf = SVC(kernel = 'rbf', C = 1, gamma =  0.01)
svm_clf.fit(X, y)
all_classifiers = [svm_vanilla, random_forest_vanilla, svm_clf, random_forest_clf]
all_classifier_names = ['svm_vanilla', 'random_forest_vanilla', 'svm_tuned', 'random_forest_tuned']
best_classifier = hold_out_test(all_classifiers, all_classifier_names)

def final_test(classifiers, classifier_names):
    df = pd.read_csv('dga.csv', index_col = False)
    df.drop(['host', 'subclass'], axis = 1, inplace = True)
    df.sample(frac=0.1, replace=True)

    df['length'] = df.apply(lambda row: len(row.domain), axis = 1)
    df['entropy'] = df['domain'].apply(lambda x : entropy(x))
    df['ngram'] = dns_counts * dns_vc.transform(df['domain']).T
    df['class'] = df['class'].apply(lambda x : 0.0 if x == 'dga' else 1)
    
    X = df.as_matrix(['length', 'entropy', 'ngram'])
    y_test = np.array(df['class'].tolist())
    for classifier, name in zip(classifiers, classifier_names): 
        y_pred = classifier.predict(X)
        print 'final test accuracy score for %s is %f' %(name, accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plot_cm(cm, ['dga', 'legit'])
    
final_test(all_classifiers, all_classifier_names)





