import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score, precision_recall_curve, average_precision_score, auc
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")

from preprocessing import prepare_train_data

def custom_train_test_split(df):
    # Split by gene_id and load train/test data into csv
    X = df.drop('label', axis = 1)
    y = df[['label']]

    gs = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
    train_ix, test_ix = next(gs.split(X, y, groups=X.gene_id))

    X_train = X.loc[train_ix]
    y_train = y.loc[train_ix]

    X_test = X.loc[test_ix]
    y_test = y.loc[test_ix]

    train_df = pd.concat([X_train,y_train],axis = 1)
    test_df = pd.concat([X_test,y_test],axis = 1)

    to_drop = ['gene_id', 'transcript_id', 'position', 'sequence']
    train = train_df.drop(to_drop, axis = 1)
    test = test_df.drop(to_drop, axis = 1)

    X_train = train.drop('label', axis = 1)
    X_test = test.drop('label', axis = 1)
    y_train = train['label']
    y_test = test['label']

    # train_df.to_csv('train.csv', index = False)
    # test_df.to_csv('test.csv', index = False)

    return X_train, y_train, X_test, y_test

def oversample_data(X_train, y_train, sampling_proportion=0.7):
    over = SMOTE(random_state = 42,sampling_strategy=sampling_proportion)
    X_train, y_train = over.fit_resample(X_train, y_train)
    return X_train, y_train

def train_rf_model(X_train, y_train):
    print('Training random forest model')
    print('Hyperparameter tuning with GridSearch')
    # hyperparameter tuning
    # rf_parameters = {'n_estimators': list(range(50,1000,50)),
    #             'max_features': list(range(50,1000,50)),
    #             'max_depth': list(range(5,100,5)),
    #             'min_samples_split': [2, 5, 10],
    #             'min_samples_leaf': [1,2,4],
    #             'bootstrap': [True, False],
    #             'random_state': [4262]}

    # rf = RandomForestClassifier(random_state = 4262)
    # best_model = GridSearchCV(rf, rf_parameters, scoring='f1_macro', verbose=0)
    # best_model.fit(X_train, y_train)

    print('Done! Fitting model with best parameters.')
    #rf = RandomForestClassifier(**best_model.best_params_)
    rf = RandomForestClassifier(random_state = 42, n_estimators = 16, max_depth = 7)
    rf = rf.fit(X_train, y_train)
    return rf


def variable_selection(X_train, y_train, var_path):
    print('Performing variable selection')
    #Adding constant column of ones, mandatory for sm.OLS model
    X_1_train = sm.add_constant(X_train)
    #Fitting sm.OLS model
    model = sm.OLS(y_train,X_1_train).fit()

    #Backward Elimination
    cols = list(X_train.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1_train = X_train[cols]
        X_1_train = sm.add_constant(X_1_train)
        model = sm.OLS(y_train,X_1_train).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>=0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols

    #Run 1 to 32 variables using rfe
    nof_list = np.arange(1,len(selected_features_BE)+1)            
    high_score = 0
    #Variable to store the optimum features
    nof = 0           
    score_list = []
    for n in range(len(nof_list)):
        model = RandomForestClassifier(random_state = 42, n_estimators = 16, max_depth = 7)
        rfe = RFE(estimator = model, n_features_to_select = nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        model.fit(X_train_rfe,y_train)
        
        predicted = model.predict(X_train_rfe)
        predicted_proba = model.predict_proba(X_train_rfe)[::,1]
        score = roc_auc_score(y_train, predicted_proba)

        score_list.append(score)
        if(score>high_score):
            high_score = score
            nof = nof_list[n]

    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))

    cols = list(X_train.columns)
    model = RandomForestClassifier(random_state = 42, n_estimators = 16, max_depth = 7)
    #Initializing RFE model
    rfe = RFE(estimator = model, n_features_to_select = nof)             
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train,y_train)  
    #Fitting the data to model
    model.fit(X_rfe,y_train)              
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    print(f'Selected features: {selected_features_rfe}')

    print(f'Saving selected features as {var_path}')
    output_features = ",".join(map(str, selected_features_rfe))
    with open(var_path, "w") as output:
        output.write(output_features)

    return selected_features_rfe


def evaluate_model(model, test, label_test):
    predicted = model.predict(test)
    predicted_proba = model.predict_proba(test)[::,1]
    precision, recall, thresholds = precision_recall_curve(label_test, predicted_proba)
    print("accuracy:", str(accuracy_score(label_test, predicted)))
    print("precision:", str(precision_score(label_test, predicted)))
    print("recall:", str(recall_score(label_test, predicted)))
    print("f1", str(f1_score(label_test, predicted)))
    print("roc auc:", str(roc_auc_score(label_test, predicted_proba)))
    print("pr auc:", str(average_precision_score(label_test, predicted_proba)))
    print('pr auc (curve)', str(auc(recall, precision)))
    classification = classification_report(label_test, predicted)
    print(classification)


def save_model(model, model_path):
    filename = model_path
    print(f'Saving Random Forest Model as {filename}')
    pickle.dump(model, open(filename, 'wb'))


def main():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--json_path', type=str, default='../data/data.json', help='Specify the path to json file here', required=True)
    parser.add_argument('--info_path', type=str, default='../data/data.info', help='Specify the path to info file here for labels', required=True)
    parser.add_argument('--data_out_path', type=str, default='../data/parsed_train_data.csv', help='Specify the path to save processed train data', required=False)
    parser.add_argument('--model_out_path', type=str, default='../model/rf_model.sav', help='Specify the path to save trained model', required=False)
    parser.add_argument('--selected_var_out_path', type=str, default='../data/selected_variables.txt', help='Specify the path to save selected variables', required=False)
    args = vars(parser.parse_args())

    df_processed = prepare_train_data(json_path=args['json_path'], info_path=args['info_path'], result_path=args['data_out_path'])

    X_train, y_train, X_test, y_test = custom_train_test_split(df_processed)

    selected_features = variable_selection(X_train, y_train, var_path=args['selected_var_out_path'])
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_selected, y_train = oversample_data(X_train_selected, y_train, 0.7)

    fitted_model = train_rf_model(X_train_selected, y_train)
    evaluate_model(model=fitted_model, test=X_test_selected, label_test=y_test)
    
    save_model(fitted_model, model_path=args['model_out_path'])

if __name__ == '__main__':
    main()