import os
import re
from io import BytesIO
from zipfile import ZipFile

import dataset
import pandas as pd
import requests

from src.tables.country import get_country_id
from src.tables.hardware import get_hardware_id
from src.tables.info_tables import create_paper_info_table, create_model_info_table
from src.tables.paper_document import create_paper_document_table
from src.tables.paper_text import create_paper_text_table

EPOCH_ZIP_URL = "http://epoch.ai/data/generated/ai_models.zip"


def extract_year(pub_date):
    if not pub_date:
        return None
    match = re.search(r"(\d{4})", str(pub_date))
    return int(match.group(1)) if match else None

def fetch_epoch_csv() -> pd.DataFrame:
    resp = requests.get(EPOCH_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with ZipFile(BytesIO(resp.content)) as z:
        csv_name = next((n for n in z.namelist() if n.lower().endswith('all_ai_models.csv')), None)
        if not csv_name:
            csv_name = next((n for n in z.namelist() if n.lower().endswith('.csv')), None)
        if not csv_name:
            raise FileNotFoundError("No CSV file found in the downloaded Epoch archive.")
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    return df.where(pd.notnull(df), None)


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names SQL-friendly (no spaces/hyphens/dots) and unique."""
    seen = {}
    new_cols = []
    for col in df.columns:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(col)).strip("_")
        safe = safe or "col"
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0
        new_cols.append(safe)
    df = df.copy()
    df.columns = new_cols
    return df


def create_epoch_table(db: dataset.Database):
    """
    Load Epoch data:
    - Save raw rows into table 'epoch'
    - Map paper fields into 'paper_info'
    - Map model fields into 'model_info'
    """
    for name in ('epoch', 'paper_info', 'model_info'):
        if name in db:
            db[name].drop()

    df = fetch_epoch_csv()

    # Raw epoch table with explicit id and safe column names
    df_epoch = _sanitize_columns(df)
    df_epoch.insert(0, 'id', range(1, len(df_epoch) + 1))
    epoch_table = db.create_table('epoch', primary_id='id', primary_type=db.types.integer)
    epoch_table.insert_many(df_epoch.to_dict(orient='records'))

    paper_info_table = create_paper_info_table(db)
    model_info_table = create_model_info_table(db)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        #if idx > 10:
        #    break


        #data = row._asdict()
        country = row['Country (of organization)']
        hardware = row['Training hardware']
        pub_year = extract_year(row['Publication date'])

        if paper_info_table.find_one(link=row['Link']) is None:
            paper_info_table.insert({
                "link": row['Link'],
                "abstract": row['Abstract'],
                "country": country,
                "id_country": get_country_id(country, db) if country else None,
                "year": pub_year,
                "split": None,
            })
        id_paper = paper_info_table.find_one(link=row['Link'])['id_paper']

        model_info_table.insert({
            "id_paper": id_paper,
            "model": row['Model'],
            "architecture": row['Task'],  # proxy: task/category as architecture
            "parameters": row['Parameters'],
            "id_hardware": get_hardware_id(hardware, db) if hardware else None,
            "hardware": hardware,
            "hardware_compute": None,
            "hardware_number": row['Hardware quantity'],
            "hardware_power": None,
            "training_compute": row['Training compute (FLOP)'],
            "training_time": row['Training time (hours)'],
            "power_draw": row['Training power draw (W)'],
            "co2eq": None,
        })

    create_paper_document_table(db)
    create_paper_text_table(db)
    


if __name__ == '__main__':
    cache_db = '.cache/tests/epoch.db'
    if os.path.exists(cache_db):
        os.remove(cache_db)
    os.makedirs('.cache/tests', exist_ok=True)
    db = dataset.connect(f'sqlite:///{cache_db}')
    create_epoch_table(db)
