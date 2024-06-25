import re
import numpy as np
import pandas as pd
from os import path, sep
from config import ROOT_PATH
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def read_date(tmstamp):
    # Rotate years
    fmt1 = '%Y-%m-%d %H:%M:%S.%f'
    fmt2 = '%d/%m/%Y %H:%M'

    try:
        tmstmp = datetime.strptime(tmstamp, fmt1)
        fmt = fmt1
    except:
        try:
            tmstmp = datetime.strptime(tmstamp, fmt2)
            fmt = fmt2
        except:
            tmstmp = datetime.strptime('2019-06-15 12:23:32.342', fmt1)
            fmt = fmt1

    return tmstmp, fmt


def printout_metrics(y, y_hat):
    # Make prediction report
    clrep = classification_report(y, y_hat, output_dict=True)
    clrep = pd.DataFrame(clrep).transpose()
    return clrep


def replace_exact_match(srs, pattern, replacement=""):

    if isinstance(pattern, list):
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, pattern)) + r")\b", flags=re.IGNORECASE)
    elif not isinstance(pattern, re.Pattern):
        raise TypeError("replace_exact_match: Forbidden words must be either list of strings or patterns")

    # Remove forbidden words from each sentence
    filtered_sentences = [pattern.sub(replacement, sentence) for sentence in srs]
    return filtered_sentences


def static_label_correction(X, out_col_service, out_col_activite, project_prefix, replacement_pattern, replacement_collapse):
    print('\tcorrecting labels ...', end='')

    # --- Remove lines not to use
    fltr = X['Line to use'] == 0
    X = X[~fltr]

    # --- Replace keys: Service
    X.loc[:, out_col_service] = X['Service ABC']
    # Rule 1: all services with ~SET_IN_ACT_KET <- KA
    fltr = X['Service ABC'] == '~SET_IN_ACT_KEY'
    X.loc[fltr, out_col_service] = X.loc[fltr, 'KA']
    # Rule 2: all non-empty KP -> Service
    fltr = ~X['KP'].isnull()
    X.loc[fltr, out_col_service] = X.loc[fltr, 'KP']

    # --- Replace Keys: Activite
    X.loc[:, out_col_activite] = X['Activite ABC']
    # Rule 5: all non-empty KA -> Activité
    fltr = ~X['KA'].isnull()
    X.loc[fltr, out_col_activite] = X.loc[fltr, 'KA']

    # Rule 3: handle project prefix
    if project_prefix == 'drop':
        X[out_col_service] = replace_exact_match(X[out_col_service], replacement_pattern, '')
    elif project_prefix == 'collapse':
        X[out_col_service] = replace_exact_match(X[out_col_service], replacement_pattern, replacement_collapse)

    # Rule 4: Stringify NaNs
    X.loc[X[out_col_activite].isna(), out_col_activite] = "nan"
    X.loc[X[out_col_service].isna(), out_col_service] = "nan"

    print('done.')
    return X


def tdfidfvectorizer(X, in_columns, out_prefix, VECTO):

    # Turn dataframe column to list
    for i_, j_, k_ in zip(in_columns, VECTO, out_prefix):
        # Extract + transform column
        i_col = X[i_]
        # Remove NaNs
        i_col_nan = i_col[~i_col.isna()]
        o_col = j_.transform(i_col_nan).toarray()

        # Put in new df
        nCols = o_col.shape[1]
        newDf = pd.DataFrame(data=np.zeros([len(X), nCols]),
                             columns=[f'{k_}_{j_}' for j_ in range(nCols)],
                             index=X.index)
        newDf.loc[i_col_nan.index, :] = o_col

        # Append new df to X
        X = pd.concat([X, newDf], axis=1)
    return X


def onehot_encoder(X, in_columns, encoders):
    # Loop on input columns
    for i_, j_ in zip(in_columns, encoders):

        # Isolate column - without nans
        i_col = X.loc[~X[i_].isna(), i_]
        i_col_exp = np.expand_dims(i_col, axis=1)
        # Encode
        i_col_enc = j_.transform(i_col_exp)
        if isinstance(i_col_enc, csr_matrix):
            i_col_enc = i_col_enc.toarray()
        # Reshape as needed
        to_exp = list(range(len(i_col_enc.shape), len(i_col_exp.shape)))
        i_col_enc = np.expand_dims(i_col_enc, to_exp)

        # Put in new df
        nCols = i_col_enc.shape[1]
        newDf = pd.DataFrame(data=np.zeros([len(X), nCols]),
                             columns=[f"{i_}_vec_{k_}" for k_ in range(nCols)],
                             index=X.index)
        newDf.loc[i_col.index, :] = i_col_enc

        # Append new df to X
        X = pd.concat([X, newDf], axis=1)

    return X


def categorical_encoder(X, in_columns, out_columns, encoders):

    # Loop on input columns
    for i_, j_, k_ in zip(in_columns, out_columns, encoders):

        # Filter labels
        i_col = X[i_]
        filter_out = np.zeros(len(X)).astype(bool)
        for l_ in pd.Series(i_col).unique():
            try:
               k_.transform([l_])
               filter_out[i_col == l_] = True
            except:
                print(f'Unknown class {l_}')

        # Make transform
        X.loc[~filter_out, j_] = -1
        X.loc[filter_out, j_] = k_.transform(i_col[filter_out])
    return X


def simple_imputer(X, in_columns, imputer):

    # Transform
    # Do imputation
    X_i = X.loc[:, in_columns].copy()
    X_i = imputer.transform(X_i)

    # Make output
    X.loc[:, in_columns] = X_i
    return X


def match_patterns_to_iterable(patterns, itr):

    FT = []
    # Loop on all patterns
    for i_ in patterns:
        # Make pattern
        ptrn = re.compile(i_)
        # Search matching words
        mtch = [word for word in itr if ptrn.search(word)]
        FT += mtch

    return FT


def agregate_minor_classes(df, cols, threshold, agregate_classes=None):
    print('\tagregating minor classes ...', end='')

    for i_ in cols:

        # Make value count
        vc = df[i_].value_counts()
        ir = vc < threshold
        agregate_classes = ir[ir].index

        # New column
        ncol = df[i_].copy()
        ncol[ncol.isin(agregate_classes)] = 'AGREGATED'
        df.loc[:, i_] = ncol.values
    print('done.')
    return df, agregate_classes


def get_stopwords_stems():
    TEXT_CLEANING = {}
    LS_NAMES = ["STOPWORDS", "STEMS"]
    LS_VARS = ["stopwords-fr", "wordstems"]
    for i_, j_ in zip(LS_NAMES, LS_VARS):
        with open((sep).join([ROOT_PATH, "data", "auxiliaire", f"{j_}.txt"]), 'r') as f:
            TEXT_CLEANING[i_] = [k_.strip() for k_ in f.readlines()[0].split(',')]
    return TEXT_CLEANING


def replace_exact_match(srs, pattern, replacement=""):

    if isinstance(pattern, list):
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, pattern)) + r")\b", flags=re.IGNORECASE)
    elif not isinstance(pattern, re.Pattern):
        raise TypeError("replace_exact_match: Forbidden words must be either list of strings or patterns")

    # Remove forbidden words from each sentence
    filtered_sentences = [pattern.sub(replacement, sentence) for sentence in srs]
    return filtered_sentences


def stem_string(sentence, stems):
    # Create a regular expression pattern for matching stems
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, stems)) + r")\w*\b", flags=re.IGNORECASE)

    return pattern.sub(
        lambda match: next(
            (prefix.lower() for prefix in stems if match.group().lower().startswith(prefix)),
            match.group().lower(),
        ),
        sentence,
    )


def recode_keys(X, params):
    print('\trecoding keys ...', end='')
    if 'key02' in X.columns:
        X.loc[:, params['UCfact_recode_keys']['output_mapping']['key02']] = X['key02'] > 500
    # Key04
    if 'key04' in X.columns:
        X.loc[:, params['UCfact_recode_keys']['output_mapping']['key02']] = X['key04'] > 500
    print('done.')
    return X


def label_project(X):
    print('\tlabelling projects ...', end='')
    matching_elements = [element for pattern in ['PRO*', 'CONSUL', 'DEVSUP'] for element in
                         X['Activite ABC'].unique() if re.search(pattern, element)]
    n_col = np.zeros(len(X))
    n_col[X['Activite ABC'].isin(matching_elements)] = 1
    X.loc[:, 'is_project'] = n_col
    print('done.')
    return X


def clean_text(X):
    print('\tCleaning text ...', end='')
    TEXT_CLEANING = get_stopwords_stems()

    X.loc[:, 'clean_text'] = X[['Objet_Facture', 'descp']].astype(str).agg(' '.join, axis=1)
    X.loc[:, 'clean_text'] = X['clean_text'].apply(lambda x: re.sub(r"[,\.;'°\+_\-/\\]", " ", x))
    X.loc[:, 'clean_text'] = X['clean_text'].apply(lambda x: x.lower())
    X.loc[:, 'clean_text'] = X['clean_text'].apply(lambda x: re.sub('[^0-9a-zA-Z ]', '', x))
    X.loc[:, 'clean_text'] = X['clean_text'].apply(lambda x: " ".join(x.split()))
    X.loc[:, 'clean_text'] = replace_exact_match(X['clean_text'], TEXT_CLEANING['STOPWORDS'])
    X.loc[:, 'clean_text'] = X['clean_text'].apply(stem_string, args=(TEXT_CLEANING['STEMS'],))
    print('done.')
    return X


def split_dates(X):
    print('\tSplitting dates ...', end='')

    X.loc[:, 'jour_facture'] = X['Date_facture'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').day)
    X.loc[:, 'mois_facture'] = X['Date_facture'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').month)
    print("done. ")
    return X


def feat_label_split(X):
    print('\tSplitting features labels ...', end='')
    y = X[match_patterns_to_iterable(['is_project', 'Activite ABC', 'Service ABC'], X.columns)]
    X = X[match_patterns_to_iterable(['Montant_Ligne_Facture', 'jour_facture', 'key02_r', 'Nom_fournisseur_vec*', 'Ordonnateur_vec*','clean_text_vec*'], X.columns)]
    print("done. ")
    return X, y