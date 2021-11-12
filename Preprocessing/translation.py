import pandas as pd
import os

from utils import translate_text

credentials_path = '../credentials/traduccion-328218-a8230e806ca8.json'
file_path = '../data/clean_text.tsv'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

data = pd.read_csv(file_path, sep='\t', lineterminator='\n')
data['translated_text'] = data['clean_text'].apply(translate_text)

data.to_csv('../data/translated_tweets.tsv', sep='\t', index=False)