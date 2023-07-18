import json

from neo4j import GraphDatabase, basic_auth


def get_connection_driver():
    with open('../credentials/neo4j_credentials.json', 'r') as f:
        credentials = json.load(f)

    driver = GraphDatabase.driver(
        credentials['url'],
        auth=basic_auth(credentials['username'], credentials['password']))

    return driver
