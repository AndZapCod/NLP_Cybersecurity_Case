import pandas as pd

from connection import get_connection_driver
from tqdm import tqdm

data = pd.read_csv('../data/clean_text.tsv', sep='\t', lineterminator='\n')

driver = get_connection_driver()

user_query = '''
MERGE (n:User {name:$name}) ON CREATE 
SET n.followers=$followers, n.friends=$friends
'''

tweet_query = '''
MERGE (n:Tweet {id:$id}) ON CREATE
SET n.text=$text, n.created_at=$date
'''

relation_query = '''
MATCH (a:User), (b:Tweet) WHERE a.name=$name AND b.id=$id MERGE (a)-[p:Post]->(b)
'''

with driver.session(database="neo4j") as session:
    for row in tqdm(data.itertuples(), 'Creating User and Tweets nodes...'):
        id = int(row[1])
        text = str(row[2])
        name = str(row[3])
        date = str(row[4])
        followers = int(row[5])
        friends = int(row[6])

        session.write_transaction(
            lambda tx: tx.run(
                user_query,
                name=name,
                followers=followers,
                friends=friends
            ).data()
        )

        session.write_transaction(
            lambda tx: tx.run(
                tweet_query,
                id=id,
                text=text,
                date=date
            ).data()
        )

        session.write_transaction(
            lambda tx: tx.run(
                relation_query,
                name=name,
                id=id
            ).data()
        )

driver.close()