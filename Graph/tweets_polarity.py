import pandas as pd

from connection import get_connection_driver
from tqdm import tqdm

data = pd.read_csv('../data/tweets_polarity.tsv', sep='\t', lineterminator='\n')

driver = get_connection_driver()

polarity_query = '''
MATCH (n:Tweet) WHERE n.id=$id SET n.polarity=$polarity
'''

with driver.session(database="neo4j") as session:
    for row in tqdm(data.itertuples(), 'Adding polarity data...'):
        id = int(row[1])
        polarity = float(row[2])

        session.write_transaction(
            lambda tx: tx.run(
                polarity_query,
                id=id,
                polarity=polarity
            ).data()
        )

driver.close()