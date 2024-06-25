import re
import pickle
import numpy as np
import pandas as pd
from os import path, sep
from copy import deepcopy
from config import ROOT_PATH
from xgboost import XGBClassifier
from datetime import datetime
from src.utils import (read_date,
                       printout_metrics,
                       static_label_correction,
                       tdfidfvectorizer,
                       onehot_encoder,
                       categorical_encoder,
                       simple_imputer,
                       match_patterns_to_iterable,
                       agregate_minor_classes,
                       replace_exact_match,
                       stem_string,
                       recode_keys,
                       label_project,
                       clean_text,
                       split_dates,
                       feat_label_split)
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




def define_defaul_params():
    allKW = {# --- PIPE DEFAULTS
             'verbose': True,
             'use_smote': True,
             'apply_modeling': True,
             'apply_lbl_encoding': True,
             'model_class': MultiOutputClassifier,
             'dimensionality_reduction': None,

             # --- BLOCK DEFAULTS
             'DataSplitter': {'test_size': 0.4,
                              'shuffle': True,
                              'random_state': 42,
                              'stratify_on': 'Service_OK'},
             'TextCleaner': {'in_columns': ['Objet_Facture', 'descp'],
                             'out_column': 'clean_text',
                             'max_digits': 10,
                             'min_charac': 1,
                             'skim_digits': True,
                             'remove_stopwords': True,
                             'remove_names': True,
                             'remove_locations': True,
                             'remove_dates': True,
                             'custom_stemming': True},
             'PCA': {'n_components': 0.95,
                     'svd_solver': 'full'},
             'UCfact_recode_keys': {'output_mapping': {'key02': 'key02_r', 'key04': 'key04_r'}},
             'UCfact_label_corrector': {'out_service': 'Service_OK',
                                        'out_activite': 'Activite_OK'},
             "UCfact_label_agregator": {'out_service': 'Service_OK',
                                        'out_activite': 'Activite_OK',
                                        'agregation_threshold': 100},
             'UCfact_label_project': {'ref_column': 'Activite_OK',
                                      'out_column': 'is_project'},
             'AWBTfidfVectorizer': {'in_columns': ['clean_text'],
                                    'out_prefix': ['clean_text_vec'],
                                    'max_features': 1000},
             'AWOneHotEncoder': {'in_columns': ['key01', 'key03', 'Nom_fournisseur', 'Ordonnateur'],
                                 'out_columns': ['key01_vec', 'key03_vec', 'Nom_fournisseur_vec', 'Ordonnateur_vec']},
             'AWCategoricalEncoderFeat': {'in_columns': ['Nom_fournisseur', 'Ordonnateur'],
                                          'out_columns': ['Nom_fournisseur_enc', 'Ordonnateur_enc']},
             'AWCategoricalEncoderLbl': {'in_columns': ['Activite_OK', 'Service_OK'],
                                         'out_columns': ['Activite_enc', 'Service_enc']},
             'UCfact_split_dates': {'in_column': 'Date_facture',
                                    'out_columns': ['jour_facture', 'mois_facture']},
             'AWSimpleImputer': {'in_columns': ['Montant_Ligne_Facture', 'jour_facture', 'mois_facture']},
             'AWBFeatureLabelSplitter': {'features': ['Montant_Ligne_Facture', 'jour_facture', 'key02_r', 'Nom_fournisseur_vec*', 'Ordonnateur_vec*', 'clean_text_vec*'],
                                         'labels': ['is_project', 'Activite_enc', 'Service_enc']},
             'SMOTE': {'row_limit': 300000,
                       'smote_on': 'Service_enc'},
             'XGBClassifier': {},
             'MultiOutputClassifier': {'estimator': XGBClassifier()},
             'LogisticRegression': {'solver': 'liblinear'},
             'ClassifierChain': {'base_estimator': SVC()},
             'RandomForestClassifier': {},
             'GradientBoostingClassifier': {},
             'TruncatedSVD': {'n_components': 50},
             'SVC': {}
             }
    return allKW


def prep_experiment():

    # Load all data
    df = pd.read_csv( path.join(ROOT_PATH, 'data', 'dummy_dataset.csv') )

    # Train <2021 /test >= 2021split
    train = df.Date_facture.apply(lambda x: read_date(x)[0].year<2020)
    df_train = df.loc[train]
    df_test = df.loc[~np.array(train)]

    return df_train, df_test


def run_pipeline(X, **kwargs):
    # --- Process keyword arguments
    # -----------------------------
    # Structure params
    defKW = define_defaul_params()
    for i_, j_ in kwargs.items():
        # Split block/params
        new_prm = i_.split('__')
        if len(new_prm)==2:
            blk, prm = new_prm
            if blk not in defKW.keys():
                defKW[blk] = {}
            defKW[blk][prm] = j_
        elif len(new_prm)==1:
            defKW[new_prm[0]] = j_

    print('\n\nRunning pipeline: ...')

    # --- Assemble pipeline
    # ---------------------
    # Recode keys
    print('\trecoding keys ...', end='')
    if 'key02' in X.columns:
        X[defKW['UCfact_recode_keys']['output_mapping']['key02']] = X['key02']>500
    # Key04
    if 'key04' in X.columns:
        X[defKW['UCfact_recode_keys']['output_mapping']['key02']] = X['key04']>500
    print('done.')

    # Correct labels
    print('\tcorrecting labels ...', end='')
    X = static_label_correction(X, 'Service ABC', 'Activite ABC', None, re.compile(r'Projets \d{4} - '), '')
    print('done.')

    # Label project
    print('\tlabelling projects ...', end='')
    matching_elements = [element for pattern in ['PRO*', 'CONSUL', 'DEVSUP'] for element in X['Activite ABC'].unique() if re.search(pattern, element)]
    n_col = np.zeros(len(X))
    n_col[X['Activite ABC'].isin(matching_elements)] = 1
    X['is_project'] = n_col
    print('done.')

    # --- Agregate
    print('\tagregating minor classes ...', end='')
    for i_ in ['Service ABC', 'Activite ABC']:
        X[i_], _ = agregate_minor_classes(X[i_], 100)
    print('done.')

    # Train/Test split
    print('\tsplitting train/test ...', end='')
    X_train, X_test = train_test_split(X, test_size=0.3, shuffle=True, stratify=X['Service ABC'], random_state=42)
    print('done.')

    # Clean text --- train
    print('\tcleaning raw text ...', end='')
    X_train['clean_text'] = X_train[['Objet_Facture', 'descp']].astype(str).agg(' '.join, axis=1)
    X_train['clean_text'] = X_train['clean_text'].apply(lambda x: re.sub(r"[,\.;'°\+_\-/\\]", " ", x))
    X_train['clean_text'] = X_train['clean_text'].apply(lambda x: x.lower())
    X_train['clean_text'] = X_train['clean_text'].apply(lambda x: re.sub('[^0-9a-zA-Z ]', '', x))
    X_train['clean_text'] = X_train['clean_text'].apply(lambda x: re.sub('[^a-z ]', '', x))
    X_train['clean_text'] = X_train['clean_text'].apply(lambda x: " ".join(x.split()))
    X_train['clean_text'] = replace_exact_match(X_train['clean_text'], TEXT_CLEANING['STOPWORDS'])
    X_train['clean_text'] = X_train['clean_text'].apply(stem_string, args=(TEXT_CLEANING['STEMS'],))

    # Clean text --- test
    X_test['clean_text'] = X_test[['Objet_Facture', 'descp']].astype(str).agg(' '.join, axis=1)
    X_test['clean_text'] = X_test['clean_text'].apply(lambda x: re.sub(r"[,\.;'°\+_\-/\\]", " ", x))
    X_test['clean_text'] = X_test['clean_text'].apply(lambda x: x.lower())
    X_test['clean_text'] = X_test['clean_text'].apply(lambda x: re.sub('[^a-z ]', '', x))
    X_test['clean_text'] = X_test['clean_text'].apply(lambda x: " ".join(x.split()))
    X_test['clean_text'] = replace_exact_match(X_test['clean_text'], TEXT_CLEANING['STOPWORDS'])
    X_test['clean_text'] = X_test['clean_text'].apply(stem_string, args=(TEXT_CLEANING['STEMS'],))
    print('done.')

    # Exclude infrequent words
    infrq = pd.Series( np.hstack(X_train['clean_text'].apply(lambda x: x.split())) ).value_counts()
    infrq = list( infrq[infrq<20].index ) + [i_ for i_ in infrq.index if len(str(i_))<3]
    X_train['clean_text'] = X_train['clean_text'].apply(lambda x: " ".join(list(set(x.split()).difference(infrq))))
    X_test['clean_text'] = X_test['clean_text'].apply(lambda x: " ".join(list(set(x.split()).difference(infrq))))

    # TD-IDF vectorizer
    print('\tvectorizing text ...', end='')
    X_train = tdfidfvectorizer(X_train, ['clean_text'], ['clean_text_vec'])
    X_test = tdfidfvectorizer(X_test, ['clean_text'], ['clean_text_vec'])
    print('done.')

    # OneHot encoder
    print('\tone-hot encoding variables ...', end='')
    X_train = onehot_encoder(X_train, ['key01', 'key03', 'Nom_fournisseur', 'Ordonnateur'])
    X_test = onehot_encoder(X_test, ['key01', 'key03', 'Nom_fournisseur', 'Ordonnateur'])
    print('done.')

    # Categorical encoder
    print('\tcategrocial variable encoding ...', end='')
    X_train = categorical_encoder(X_train, ['Activite ABC', 'Service ABC'], ['Activite ABC', 'Service ABC'])
    X_test = categorical_encoder(X_test, ['Activite ABC', 'Service ABC'], ['Activite ABC', 'Service ABC'])
    print('done.')

    # Split dates Jour/Mois
    print('\tsplitting dates ...', end='')
    X_train.loc[:, 'jour_facture'] = X_train['Date_facture'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').day)
    X_test.loc[:, 'jour_facture'] = X_test['Date_facture'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').day)
    X_train.loc[:, 'mois_facture'] = X_train['Date_facture'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').month)
    X_test.loc[:, 'mois_facture'] = X_test['Date_facture'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').month)
    print('done.')

    # Imputation
    print('imputing missing values ...', end='')
    X_train = simple_imputer(X_train, ['Montant_Ligne_Facture', 'jour_facture', 'mois_facture'])
    X_test = simple_imputer(X_test, ['Montant_Ligne_Facture', 'jour_facture', 'mois_facture'])
    print('done.')

    # Feature/Label splitter
    print('\tfeature/label splitting ...', end='')
    y_train = X_train[match_patterns_to_iterable(['is_project', 'Activite ABC', 'Service ABC'], X_train.columns)]
    y_test = X_test[match_patterns_to_iterable(['is_project', 'Activite ABC', 'Service ABC'], X_test.columns)]
    X_train = X_train[match_patterns_to_iterable(['Montant_Ligne_Facture', 'jour_facture', 'key02_r', 'Nom_fournisseur_vec*', 'Ordonnateur_vec*', 'clean_text_vec*'], X_train.columns)]
    X_test = X_test[match_patterns_to_iterable(['Montant_Ligne_Facture', 'jour_facture', 'key02_r', 'Nom_fournisseur_vec*', 'Ordonnateur_vec*', 'clean_text_vec*'], X_test.columns)]
    print('done.')

    # Standard scaler
    print('\tstandard scaler ...', end='')
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    print('done.')

    # Modelling
    mdl = MultiOutputClassifier(estimator=XGBClassifier())
    mdl.fit(X_train, y_train.astype(int))

    # --- Train pipeline
    # ------------------
    y_pred_tr = mdl.predict(X_train)
    y_pred_vl = mdl.predict(X_test)

    return y_train, y_pred_tr, y_test, y_pred_vl


def run_eval_pipeline(df, **kwargs):
    # Compile and run
    pipeline, y_train, y_pred_tr, y_valid, y_pred_vl, classes = run_pipeline(df, **kwargs)

    print(f"Length of training data: {len(y_train)} rows")
    print(f"Length of validation data: {len(y_valid)} rows")

    # Metrics
    ev_train = printout_metrics(y_train, y_pred_tr, classes)
    ev_valid = printout_metrics(y_valid, y_pred_vl, classes)
    return ev_train, ev_valid


class Pipeline(object):

    def __init__(self, params):
        self.params = params

    def fit(self, X):
        # --- Assemble pipeline
        # ---------------------
        # Recode keys
        X = recode_keys(X, self.params)

        # Correct labels
        X = static_label_correction(X, 'Service ABC', 'Activite ABC', None, re.compile(r'Projets \d{4} - '), '')

        # Label project
        X = label_project(X)

        # Agregate
        X, _ = agregate_minor_classes(X, ['Service ABC', 'Activite ABC'], 100)

        # Clean text --- train
        X = clean_text(X)

        # Train/Test split
        X_train, X_test = train_test_split(X, test_size=0.3, shuffle=True, stratify=X['Service ABC'], random_state=42)

        # Exclude infrequent words
        infrq = pd.Series(np.hstack(X_train['clean_text'].apply(lambda x: x.split()))).value_counts()
        infrq = list(infrq[infrq < 20].index) + [i_ for i_ in infrq.index if len(str(i_)) < 3]
        X_train.loc[:, 'clean_text'] = X_train['clean_text'].apply(lambda x: " ".join(list(set(x.split()).difference(infrq))))

        # TD-IDF vectorizer
        print('\tTD-IDF vectorizer ...', end='')
        self.vectorizer = TfidfVectorizer(use_idf=True)
        self.vectorizer.fit(X_train.loc[~X_train['clean_text'].isna(), 'clean_text'])
        X_train = tdfidfvectorizer(X_train, ['clean_text'], ['clean_text_vec'], [self.vectorizer])
        print('done.')

        # OneHot encoder
        print('\tOneHot encoder ...', end='')
        in_columns = ['key01', 'key03', 'Nom_fournisseur', 'Ordonnateur']
        self.hotencoders = [OneHotEncoder(handle_unknown='ignore') for i_ in in_columns]
        for i_,j_ in zip(in_columns, self.hotencoders):
            j_.fit(np.expand_dims(X_train.loc[~X_train[i_].isna(), i_], axis=1))
        X_train = onehot_encoder(X_train, in_columns, self.hotencoders)
        print('done.')

        # Categorical encoder
        print('\tCategorical encoder ...', end='')
        in_columns = ['Activite ABC', 'Service ABC']
        self.catencoders = [LabelEncoder() for i_ in in_columns]
        for i_,j_ in zip(in_columns, self.catencoders):
            j_.fit(np.expand_dims(X_train.loc[~X_train[i_].isna(), i_], axis=1))
        X_train = categorical_encoder(X_train, in_columns, in_columns, self.catencoders)
        print('done.')

        # Split dates Jour/Mois
        X_train = split_dates(X_train)

        # Imputation
        print('\tImputation ...', end='')
        in_columns = ['Montant_Ligne_Facture', 'jour_facture', 'mois_facture']
        self.imputer = SimpleImputer()
        self.imputer.fit(X_train[in_columns], None)
        X_train = simple_imputer(X_train, in_columns, self.imputer)
        print('done.')

        # Feature/Label splitter
        X_train, y_train = feat_label_split(X_train)

        # Standard scaler
        print('\tstandard scaler ...', end='')
        self.scl = StandardScaler()
        X_train = self.scl.fit_transform(X_train)
        print('done.')

        # Modelling
        self.mdl = MultiOutputClassifier(estimator=XGBClassifier())
        self.mdl.fit(X_train, y_train.astype(int))
        return X, X_test


    def predict(self, X):
        # --- Assemble pipeline
        # ---------------------
        # Recode keys
        X = recode_keys(X, self.params)

        # Correct labels
        X = static_label_correction(X, 'Service ABC', 'Activite ABC', None, re.compile(r'Projets \d{4} - '), '')

        # Label project
        X = label_project(X)

        # Agregate
        X, _ = agregate_minor_classes(X, ['Service ABC', 'Activite ABC'], 100)

        # Clean text --- train
        X = clean_text(X)

        # Exclude infrequent words
        infrq = pd.Series(np.hstack(X['clean_text'].apply(lambda x: x.split()))).value_counts()
        infrq = list(infrq[infrq < 20].index) + [i_ for i_ in infrq.index if len(str(i_)) < 3]
        X['clean_text'] = X['clean_text'].apply(lambda x: " ".join(list(set(x.split()).difference(infrq))))

        # TD-IDF vectorizer
        X = tdfidfvectorizer(X, ['clean_text'], ['clean_text_vec'], [self.vectorizer])

        # OneHot encoder
        in_columns = ['key01', 'key03', 'Nom_fournisseur', 'Ordonnateur']
        X = onehot_encoder(X, in_columns, self.hotencoders)

        # Categorical encoder
        in_columns = ['Activite ABC', 'Service ABC']
        X = categorical_encoder(X, in_columns, in_columns, self.catencoders)

        # Split dates Jour/Mois
        X = split_dates(X)

        # Imputation
        in_columns = ['Montant_Ligne_Facture', 'jour_facture', 'mois_facture']
        X = simple_imputer(X, in_columns, self.imputer)

        # Feature/Label splitter
        X, y = feat_label_split(X)

        # Standard scaler
        X = self.scl.transform(X)

        # --- Train pipeline
        # ------------------
        y_pred = self.mdl.predict(X)
        return y, y_pred


class Experiment(object):

    def __init__(self, name, df_train, df_test, pipe_type='full', run_eval=True, **kwargs):
        # Internal params
        self.name = name
        self.params = kwargs
        self.pipeline = Pipeline(kwargs)
        self.run_eval = run_eval
        self.pipe_type = pipe_type

        # Run training/inference
        self.X_train, self.X_valid = self.pipeline.fit(df_train)
        self.y_tr_true, self.y_tr_pred = self.pipeline.predict(self.X_train)
        self.y_vl_true, self.y_vl_pred = self.pipeline.predict(self.X_valid)
        self.y_ts_true, self.y_ts_pred = self.pipeline.predict(df_test)

        # Make sure shapes are OK
        self.y_tr_pred = self.y_tr_pred.reshape(self.y_tr_true.shape)
        self.y_vl_pred = self.y_vl_pred.reshape(self.y_vl_true.shape)
        self.y_ts_pred = self.y_ts_pred.reshape(self.y_ts_true.shape)

        # Make sure type is OK
        self.y_tr_pred = self.y_tr_pred.astype( type(np.ravel( self.y_tr_true )[0]) )
        self.y_vl_pred = self.y_vl_pred.astype( type(np.ravel( self.y_vl_true )[0]) )
        self.y_ts_pred = self.y_ts_pred.astype( type(np.ravel( self.y_ts_true )[0]) )

        if isinstance(self.y_tr_true, pd.Series) or isinstance(self.y_tr_true, pd.DataFrame):
            self.y_tr_true = self.y_tr_true.values
        if isinstance(self.y_vl_true, pd.Series) or isinstance(self.y_vl_true, pd.DataFrame):
            self.y_vl_true = self.y_vl_true.values
        if isinstance(self.y_ts_true, pd.Series) or isinstance(self.y_ts_true, pd.DataFrame):
            self.y_ts_true = self.y_ts_true.values

        # Determine classification type
        labels = kwargs['AWBFeatureLabelSplitter']['labels']

        # Check evals
        if self.run_eval:

            self.ev_train = []
            self.ev_valid = []
            self.ev_test = []

            for x_, i_ in enumerate(labels):
                # Train
                y_true, y_pred = self.y_tr_true[:, x_], self.y_tr_pred[:, x_]
                self.ev_train.append(printout_metrics(y_true.astype(int), y_pred))
                # Valid
                y_true, y_pred = self.y_vl_true[:, x_], self.y_vl_pred[:, x_]
                self.ev_valid.append(printout_metrics(y_true.astype(int), y_pred))
                # Test
                #y_true, y_pred = self.y_test[:, x_], self.y_pred_ts[:, x_]
                #self.ev_test.append(printout_metrics(y_true, y_pred))


    def print(self):
        lsCols = ['accuracy', 'macro avg', 'weighted avg']

        print('\nTraining results:')
        for x_, i_ in enumerate(self.ev_train):
            print(f'\tLevel {x_}')
            print(i_.loc[lsCols, :])
            print('')

        print('\n\nValidation results:')
        for x_, i_ in enumerate(self.ev_valid):
            print(f'\tLevel {x_}')
            print(i_.loc[lsCols, :])

        #print('\n\nTest results:')
        #for x_, i_ in enumerate(self.ev_test):
        #    print(f'\tLevel {x_}')
        #    print(i_.loc[lsCols, :])
        return self

    def save(self, pipetype='full'):
        # Generate save path
        tmnow = datetime.now()
        tmstamp = f"{tmnow.day}-{tmnow.month}-{tmnow.year}_{tmnow.hour}h{tmnow.minute}m{tmnow.second}s"

        # Prep scores
        scr = self.ev_valid[-1].loc[ ['macro avg', 'weighted avg'], 'f1-score']

        # Save Experiment
        savep = sep.join([ROOT_PATH, "models", "Experiment_%s_%s_test_macro_%.3f_wavg_%.3f.pkl"]) % (pipetype, tmstamp, scr['macro avg'], scr['weighted avg'])
        with open(savep, 'wb') as f:
            pickle.dump(self, f)

        # Save Pipeline
        savep = sep.join([ROOT_PATH, "models", "Pipeline_%s_%s_test_macro_%.3f_wavg_%.3f.pkl"]) % (
        pipetype, tmstamp, scr['macro avg'], scr['weighted avg'])
        with open(savep, 'wb') as f:
            pickle.dump(self.pipeline, f)


    def load(self, filename):
        # Make path
        loadp = sep.join([ROOT_PATH, 'models', filename])

        # Save pipeline
        with open(loadp, 'rb') as f:
            self = pickle.load(f)
        return self


if __name__ == '__main__':
    # -- Prep data
    # ------------

    # Load data
    df_train, df_test = prep_experiment()

    # Get hyperp values
    param = define_defaul_params()

    # Launch experiment
    expe = Experiment('Vanilla', df_train, df_test, run_eval=True, **param)
    expe.print()
    expe.save(pipetype=False)
