from connection import get_connection_driver

driver = get_connection_driver()

tweet_constrain_query = '''
CREATE CONSTRAINT ON (n:Tweet) ASSERT n.id IS UNIQUE
'''

user_constrain_query = '''
CREATE CONSTRAINT ON (n:User) ASSERT n.name IS UNIQUE
'''

with driver.session(database="neo4j") as session:
    session.write_transaction(
        lambda tx: tx.run(
            tweet_constrain_query
        ).data()
    )

    session.write_transaction(
        lambda tx: tx.run(
            user_constrain_query
        ).data()
    )