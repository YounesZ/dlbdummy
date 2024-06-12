import numpy as np
import pandas as pd
from os import sep
from config import ROOT_PATH
from datetime import timedelta, datetime
from unidecode import unidecode
from src.utils import read_date


ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'


def random_strings(n_strings, is_capital):
    out = ''
    for i_ in range(n_strings):
        out += ALPHABET[np.random.choice(26)]

    # Apply upper
    if not is_capital:
        out = out.lower()
    return out


def swap_digits_letters(string, stt=30):
    assert isinstance(string, str)
    out = ''
    for i_ in string:
        if i_ in ALPHABET:
            out += str(ALPHABET.index(i_) % 10)
        elif i_ in DIGITS:
            #inc = np.random.randint(26)
            out += ALPHABET[(stt + int(i_)) % 26]
            #stt += inc
        else:
            out += i_
    return out


def rotate_date(tmstamp):

    # Read date
    tmstmp, fmt = read_date(tmstamp)

    # Add random years
    n_years = ( np.random.choice(4)-2 ) * 365
    n_months= (np.random.choice(13) - 6) / 12 * 30
    n_days = np.random.choice(31) - 15
    tmstmp += timedelta(days = n_days + n_months + n_years)

    return tmstmp.strftime(fmt)


def gen_value_from_histogram(X, Y):
    xx = (X + np.append(X[1:], X[-1] * 2)) / 2
    rndv = np.random.random(len(Y))
    yy = Y * rndv * Y / np.sum(Y) / np.sum(rndv)
    return np.sum(yy)


def gen_sentences():
    with open( sep.join([ROOT_PATH, 'data', 'lorem.csv']), 'r') as f:
        allS = f.readlines()
        allS = [i_.split('.') for i_ in allS]
        allS = [i_ for i_ in allS if '\n' not in i_]
        allS = [i_.strip() for i_ in sum(allS, [])]
    return allS


def recode_column(dtf, colname, separ):
    allO = [i_.strip().split(separ) for i_ in dtf.loc[~dtf[colname].isna(), colname].unique()]
    allO = sum(allO, [])
    allO = [i_.split(' ') for i_ in allO]
    allO = np.unique( sum(allO, []) )
    allO = [unidecode(i_) for i_ in allO if len(i_)>=3]
    return allO


def realistic_dummy(n_lines):

    # Load original data
    df = pd.read_csv('C:\\Users\\you.zerouali\\Documents\\Code\\classif-factures\\data\\Standard\\factures_all_years.csv', sep=";", encoding='latin1')

    # Activite ABC
    df.loc[df['Activite ABC'] == '0', 'Activite ABC'] = 'ISGOVE'
    df.loc[~df['Activite ABC'].isna(), 'Activite ABC'] = df.loc[~df['Activite ABC'].isna(), 'Activite ABC'].apply(lambda x: x[:3] + random_strings(3, True))

    # Cde_fournisseur
    df['Cde_fournisseur'] = df['Cde_fournisseur'].astype(str).apply(lambda x: swap_digits_letters(x))

    # Code_Utilisateur
    df['Code_Utilisateur'] = df['Code_Utilisateur'].astype(str).apply(lambda x: swap_digits_letters(x))

    #  'Date_Comptable'/'Date_paiement'/'Date_facture'
    df['Date_Comptable'] = df['Date_Comptable'].apply(lambda x: rotate_date(x))
    df['Date_Paiement'] = df['Date_Paiement'].apply(lambda x: rotate_date(x))
    df['Date_facture'] = df['Date_facture'].apply(lambda x: rotate_date(x))

    # KA/KP
    df.loc[~df['KA'].isna(), 'KA'] = df.loc[~df['KA'].isna(), 'KA'].astype(str).apply(lambda x: swap_digits_letters(x))
    df.loc[~df['KP'].isna(), 'KP'] = df.loc[~df['KP'].isna(), 'KP'].astype(str).apply(lambda x: swap_digits_letters(x))

    # Libelle
    df = df.drop(columns=['Libelle'])

    # Line to use
    df['Line to use'] = (np.random.random(len(df))<df['Line to use'].mean()).astype(int)

    # Montant_Ligne_Facture
    yy, xx = np.histogram(np.abs(df.loc[~df['Montant_Ligne_Facture'].isna(), 'Montant_Ligne_Facture']))
    df['Montant_Ligne_Facture'] = df['Montant_Ligne_Facture'].apply(lambda x: gen_value_from_histogram(xx, yy))

    # No_Facture, No_Ligne_Facture
    df = df.drop(columns=['No_Facture'])
    df = df.drop(columns=['No_Ligne_Facture'])

    # Nom fournisseur
    allF = [i_.split() for i_ in list( df.loc[~df['Nom_fournisseur'].isna(), 'Nom_fournisseur'].unique() )]
    allF = [i_ for i_ in sum(allF, []) if len(i_)>2]
    nval = [np.random.choice(df['Nom_fournisseur'].nunique()-1) for i_ in range(len(df))]
    allF = [f"{allF[j_]} {allF[j_+1]}" for i_,j_ in zip(range(len(df)), nval)]
    df['Nom_fournisseur'] = allF

    # Objet Facture
    allS = gen_sentences()
    df['Objet_Facture'] = [allS[i_ % len(allS)] for i_ in range(len(df))]

    # Ordonnateur
    allO = recode_column(df, 'Ordonnateur', '_')
    allO = [f"{allO[np.random.choice(len(allO))]}_{allO[np.random.choice(len(allO))]}_{allO[np.random.choice(len(allO))]}" for i_ in range(df.Ordonnateur.nunique())]
    df['Ordonnateur'] = [np.random.choice(allO) for i_ in range(len(df))]
    df['Ordonnateur']

    # Service ABC
    allSrv = recode_column(df, 'Service ABC', '-')
    unksrv = list( df['Service ABC'].unique() )
    allSrv = [f"{np.random.choice(allSrv)} {np.random.choice(allSrv)}" for i_ in range(len(unksrv))]
    df['Service ABC'] = [allSrv[unksrv.index(i_)] for i_ in df['Service ABC']]

    # Statut_Facture, Unnamed: 0
    df = df.drop(columns=['Statut_Facture'])
    df = df.drop(columns=['Unnamed: 0'])

    # descp
    allS = gen_sentences()
    df['descp'] = [allS[i_ % len(allS)] for i_ in range(len(df))]

    # 'key01', 'key02', 'key03', 'key04'
    rnd = [np.random.randint(1000) for i_ in range(4)]
    df['key01'] = df['key01'].apply(lambda x: (x * rnd[0]) % 1432)
    df['key02'] = df['key02'].apply(lambda x: (x * rnd[1]) % 1633)
    df['key03'] = df['key03'].apply(lambda x: (x * rnd[2]) % 1726)
    df['key04'] = df['key04'].apply(lambda x: (x * rnd[3]) % 1128)

    # Save
    pth = sep.join([ROOT_PATH, 'data', 'dummy_dataset.csv'])
    df[:n_lines].to_csv(pth)


if __name__ == '__main__':
    realistic_dummy(50000)