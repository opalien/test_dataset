import os
import dataset
import pandas as pd
from rapidfuzz.distance import JaroWinkler

COUNTRY_URL = "https://ourworldindata.org/grapher/carbon-intensity-electricity.csv"

def _populate_country_table(table: "dataset.Table"):
    df = pd.read_csv(COUNTRY_URL, storage_options={'User-Agent': 'Mozilla/5.0'})
    df_latest = df.sort_values('Year').drop_duplicates('Entity', keep='last')
    table.insert_many([{"name": row['Entity'], "carbon_intensity": row['Carbon intensity of electricity - gCO2/kWh']} for _, row in df_latest.iterrows()])

def create_country_table(db: dataset.Database):
    table = db.create_table('country', primary_id='id_country', primary_type=db.types.integer)
    table.create_column('name', db.types.string)
    table.create_column('carbon_intensity', db.types.float)
    _populate_country_table(table)


def get_country_id(country_name: str, db: dataset.Database):
    if not country_name:
        return None
    country_name = country_name.split(',')[0].strip()

    if 'country' not in db.tables:
        create_country_table(db)

    table = db['country']
    if table.count() == 0:
        _populate_country_table(table)

    countries = list(table)
    if not countries:
        return None

    # Argmin on the normalized distance to find the closest country name
    best_match = min(countries, key=lambda row: JaroWinkler.normalized_distance(country_name, row['name']))
    return best_match['id_country']

    #for _, row in db['country'].iterrows():
    #    score = JaroWinkler.normalized_similarity(country_name, row['name'])
    #    if score < min_score:
    #        min_score = score
    #        min_id = row['id_country']
    #return min_id


if __name__ == '__main__':
    if os.path.exists('.cache/tests/country.db'):
        os.remove('.cache/tests/country.db')
    os.makedirs('.cache/tests', exist_ok=True)
    db = dataset.connect('sqlite:///.cache/tests/country.db')
    create_country_table(db)