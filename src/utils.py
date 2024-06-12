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
from sklearn.feature_extraction.text import TfidfVectorizer



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
    # --- Remove lines not to use
    fltr = X['Line to use'] == 0
    X = X[~fltr]

    # --- Replace keys: Service
    X[out_col_service] = X['Service ABC']
    # Rule 1: all services with ~SET_IN_ACT_KET <- KA
    fltr = X['Service ABC'] == '~SET_IN_ACT_KEY'
    X.loc[fltr, out_col_service] = X.loc[fltr, 'KA']
    # Rule 2: all non-empty KP -> Service
    fltr = ~X['KP'].isnull()
    X.loc[fltr, out_col_service] = X.loc[fltr, 'KP']

    # --- Replace Keys: Activite
    X[out_col_activite] = X['Activite ABC']
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
    return X


def tdfidfvectorizer(X, in_columns, out_prefix):
    VECTO = []
    for i_ in in_columns:
        VECTO.append(TfidfVectorizer(use_idf=True))

    for i_, j_ in zip(in_columns, VECTO):
        # Extract + transform column
        i_col = X[i_]
        # Remove NaNs
        i_col_nan = i_col[~i_col.isna()]
        j_.fit(i_col_nan)

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


def onehot_encoder(X, in_columns):
    encoders = [OneHotEncoder(handle_unknown='ignore') for i_ in in_columns]

    # Loop on input columns
    for i_, j_ in zip(in_columns, encoders):
        # Isolate column - without nans
        i_col = X.loc[~X[i_].isna(), i_]
        i_col_exp = np.expand_dims(i_col, axis=1)
        # Encode
        j_.fit(i_col_exp)

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


def categorical_encoder(X, in_columns, out_columns):
    encoders = [LabelEncoder() for i_ in in_columns]

    for i_, j_ in zip(in_columns, encoders):
        # Isolate column - without nans
        i_col = X.loc[~X[i_].isna(), i_]
        i_col_exp = np.expand_dims(i_col, axis=1)
        # Encode
        j_.fit(i_col_exp)

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


def simple_imputer(X, in_columns):
    # Instantiate
    imputer = SimpleImputer()

    # Fit
    X_i = X.loc[:, in_columns].copy()
    imputer.fit(X_i, None)

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


def agregate_minor_classes(col, threshold, agregate_classes=None):
    if agregate_classes is None:
        # Make value count
        vc = col.value_counts()
        ir = vc < threshold
        agregate_classes = ir[ir].index

    # New column
    ncol = col.copy()
    ncol[ncol.isin(agregate_classes)] = 'AGREGATED'
    return ncol, agregate_classes


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
