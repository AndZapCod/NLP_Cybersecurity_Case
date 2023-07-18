import pandas as pd

from connection import get_connection_driver
from tqdm import tqdm

data = pd.read_csv('../data/tweets_clusters.tsv', sep='\t', lineterminator='\n')

driver = get_connection_driver()

cluster_query = '''
MATCH (n:Tweet) WHERE n.id=$id SET n.cluster=$cluster 
'''

with driver.session(database="neo4j") as session:
    for row in tqdm(data.itertuples(), 'Adding cluster data...'):
        id = int(row[1])
        cluster = int(row[2])

        session.write_transaction(
            lambda tx: tx.run(
                cluster_query,
                id=id,
                cluster=cluster
            ).data()
        )

driver.close()