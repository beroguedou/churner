import os 
import pickle
import sklearn
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def loader(datapath):
    return pd.read_csv(datapath)

def splitter(df, list_categorical, list_numerical, label, seed, percentage=0.2):
    df = df[list_categorical+list_numerical+[label]]
    df_train, df_test = train_test_split(df, test_size=percentage, random_state=seed, stratify=df[label])
    return df_train, df_test

def formater(df, list_variables, new_type):
    df[list_variables] = df[list_variables].astype(new_type)
    return df

def nan_replacer(df, list_variables, list_symbols):
    for symbol in list_symbols:
        df[list_variables] = df[list_variables].replace(symbol, np.nan)
    return df

def encoder(df, list_categorical, label, savepath=".", option=None):
    option = 'train' if option is None else option 

    if option == 'inference':
        list_to_encode = list_categorical 
    else:
        list_to_encode = list_categorical + [label]
        
    for var in list_to_encode:
        name = '{}.pkl'.format(var.lower())
        var_savepath = os.path.join(savepath, name)
        if option == 'train':
            # instancier un label encodeur
            le = sklearn.preprocessing.LabelEncoder()
            # Entrainer le label encodeur
            le.fit(df[var])
            # Sauvegarder le label encodeur
            with open(var_savepath, 'wb') as file:
                pickle.dump(le, file)
        else:
            # Charger le label encodeur
            with open(var_savepath, 'rb') as file:
                le  = pickle.load(file)
        # Transformer la colonne avec le label encodeur entrainé préalablement
        df[var] = le.transform(df[var])

    print("======= L'encodage des variables catégorielles est terminée ! =======")
    return df

def imputer_median(X, var_names, col='TotalCharges', savepath="."):
    id_ToCh = var_names.index(col)
    col_values = X[:, id_ToCh]
    name = "{}_median.pkl".format(col)
    var_savepath = os.path.join(savepath, name)
    
    # Compute the median
    median = np.nanmedian(col_values)
    # Save the median as an artifact
    with open(var_savepath, 'wb') as file:
        pickle.dump(median, file)

    return X


def preprocessor(df, config, option_output=None, option_train=None):
    df = df.copy()
    option_output = 'all' if option_output is None else option_output
    option_train = 'train' if option_train is None else option_train
    if option_train == 'train':
        # Suppression des doublons
        df = df.drop_duplicates()
    # Remplacement des valeurs symbols par nan
    df = nan_replacer(df, list_variables=config['data']['numerical'], list_symbols=[" "])
    # Conversion des variables numériques au bon format
    df = formater(df, list_variables=config['data']['numerical'], new_type='float32')
    # Conversion des variables catégorielles au bon format
    df = formater(df, list_variables=config['data']['categorical'], new_type='object')
    # Encoder les variables catégorielles
    df = encoder(df,
                 list_categorical=config['data']['categorical'],
                 label=config['data']['label'], 
                 savepath=config['model']['savepath'],
                 option=option_train)
    # Separate label and other variables
    df = df.reset_index(drop=True)
    var_names = config['data']['numerical']+config['data']['categorical']
    X = df[var_names].values
    if option_train == 'train':
        # Impute the median for TotalCharges
        X = imputer_median(X,
                           var_names,
                           col='TotalCharges', 
                           savepath=config['model']['savepath']
                          )
        
    if option_output == 'all':
        y = df[config['data']['label']].values
        output = X, y
    if option_output == 'inference':
        output = X
        
    return output


def custom_cross_val_score(max_depth, n_estimators, eta, subsample, cv, seed, config, X, y):

    scores = []
    skf = StratifiedKFold(n_splits=config['model']['k'], 
                      random_state=config['seed'], 
                      shuffle=True).split(X, y)

    for train_idx, val_idx in skf:
        # Définition du jeu d'entrainement et de validation
        # pour la passe courante
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        # Calcul de la median de la colonne TotalCharges
        var_names = config['data']['categorical']+config['data']['numerical']
        id_ToCh = var_names.index('TotalCharges')
        median = np.nanmedian(X_train[:, id_ToCh])
        col_values = X_train[:, id_ToCh]
        col_values[np.isnan(col_values)] = median
        X_train[:, id_ToCh] = col_values
        # Instanciation
        xgb_clf = XGBClassifier(random_state=seed,
                                max_depth=max_depth,
                                eta=eta,
                                subsample=subsample,
                                n_estimators=n_estimators,
                                use_label_encoder=False,
                                eval_metric='auc', 
                                n_jobs=8)
        # Entrainement
        xgb_clf.fit(X_train, y_train)
        # Prediction
        col_values = X_val[:, id_ToCh]
        col_values[np.isnan(col_values)] = median
        X_val[:, id_ToCh] = col_values
        y_val_pred = xgb_clf.predict(X_val)
        # Métrique ou performance
        score = sklearn.metrics.roc_auc_score(y_val, y_val_pred)
        scores.append(score)


    # Métrique
    score = np.mean(scores)

    return score


def objective(trial, config, X, y):
    # Définition des hypar-paramètres
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 4, 32)
    eta = trial.suggest_float("eta", 0, 1)
    subsample = trial.suggest_float("subsample", 0, 1)
    # Kfold training
    score = custom_cross_val_score(max_depth, 
                                   n_estimators, 
                                   eta=eta, 
                                   subsample=subsample, 
                                   cv=5, 
                                   seed=42, 
                                   config=config, 
                                   X=X, 
                                   y=y
                                  )
    return score

def toto():
    return 'yes'

    