import pandas as pd

from connection import get_connection_driver
from tqdm import tqdm

data = pd.read_csv('../data/edges_tinfoleak.csv', sep=',',
                   lineterminator='\n', skiprows=1, header=None)

driver = get_connection_driver()

user_query = '''
MERGE (n:User {name:$name})
'''

relation_query = '''
MATCH (a:User), (b:User)
WHERE a.name=$follower AND b.name=$user
MERGE (a)-[r:Follows]->(b)
'''


with driver.session(database="neo4j") as session:
    for row in tqdm(data.itertuples(), 'Creating User and Tweets nodes...'):
        user = row[1]
        follower = row[2]

        session.write_transaction(
            lambda tx: tx.run(
                user_query,
                name=user,
            ).data()
        )

        session.write_transaction(
            lambda tx: tx.run(
                user_query,
                name=follower,
            ).data()
        )

        session.write_transaction(
            lambda tx: tx.run(
                relation_query,
                follower=follower,
                user=user
            ).data()
        )

driver.close()