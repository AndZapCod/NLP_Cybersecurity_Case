import pandas as pd
import pickle

from connection import get_connection_driver
from tqdm import tqdm

with open('../data/ner_results.pkl', 'rb') as f:
    data = pickle.load(f)

driver = get_connection_driver()

entity_query = '''
MERGE (n:Entity {type:$type, token:$token}) 
'''

relation_query = '''
MATCH (a:Tweet), (b:Entity) 
WHERE a.id=$id AND b.type=$type AND b.token=$token 
MERGE (b)-[r:In]->(a)
'''

with driver.session(database="neo4j") as session:
    for row in tqdm(data.itertuples(), 'Creating Entity nodes...'):
        id = int(row[1])
        tokens = row[2]
        predictions = row[4]
        for tok, pred in zip(tokens, predictions):
            if pred != 'O':
                pred = str(pred[2:])
                tok = str(tok)

                session.write_transaction(
                    lambda tx: tx.run(
                        entity_query,
                        type=pred,
                        token=tok
                    ).data()
                )

                session.write_transaction(
                    lambda tx: tx.run(
                        relation_query,
                        id=id,
                        type=pred,
                        token=tok
                    ).data()
                )

driver.close()