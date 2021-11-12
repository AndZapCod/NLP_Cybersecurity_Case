import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.dates import DateFormatter
from utils import str_to_date
from utils import text_processing
from utils import empty_str_to_na
from datetime import datetime

file_path = '../data/Ecuador_26-27Oct2021.tsv'

data = pd.read_csv(file_path, sep='\t', lineterminator='\n')[
    [
        'id_str', 'text', 'from_user', 'created_at',
        'user_followers_count', 'user_friends_count'
    ]
]

data['created_at'] = data['created_at'].apply(str_to_date)
# Tweets created after 25 Oct 2021 at 19:00 UTC
# Tweets created after 26 Oct 2021 at 00:00 GMT-5 Quito hour
filter_date = datetime(2021, 10, 25, 19).date()
data = data[
    data['created_at'].apply(datetime.date) >= filter_date].reset_index(
    drop=True)
# Clean text
data['clean_text'] = data['text'].apply(text_processing)
data['clean_text'] = data['clean_text'].apply(empty_str_to_na)
# Remove noisy data
data = data.dropna()
data = data.drop_duplicates()

start = data['created_at'].min()
end = data['created_at'].max()
# Report
print(f'Data obtained between {start} and {end}')
print(data.info())

timeindex = pd.DatetimeIndex(data['created_at'])
timeserie = pd.Series([1] * len(data['created_at']), index=timeindex)
timeserie = timeserie.resample('H').count()

sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

formater = DateFormatter('%d-%H')
ax.xaxis.set_major_formatter(formater)
plot = sns.lineplot(data=timeserie, ax=ax)
plot.set(xlabel='Creation time', ylabel='Number of tweets')
fig.savefig('Img/timeserie.pdf', format='pdf')

# Save results
data.to_csv('../data/clean_text.tsv', sep='\t', index=False)