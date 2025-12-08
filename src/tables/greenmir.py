import os
import dataset
import pandas as pd

from src.tables.country import get_country_id
from src.tables.hardware import get_hardware_id
from src.tables.paper_document import create_paper_document_table
from src.tables.info_tables import create_paper_info_table, create_model_info_table
from src.tables.paper_text import create_paper_text_table



GREENMIR_PATH = "static/dataset.csv"
#PAPER_COLUMNS = ['id_paper','link','abstract','country_of_organization','year','id_country','split']
# dataset columns: model name	link	abstract	country	year	parameters	hardware	hardware_compute	hardware_number	hardware_power	training_compute	training_time	power_draw	co2eq												

#MODEL_COLUMNS = [
#    'id_model',
#    'id_paper',
#    'model',
#    'architecture',
#    'parameters',
#    'id_hardware',
#    'hardware_compute',
#    'hardware_number',
#    'hardware_power',
#    'training_compute',
#    'training_time',
#    'power_draw',
#    'co2eq',
#]

def create_greenmir_table(db: dataset.Database):

    df = pd.read_csv(GREENMIR_PATH)
    
    # Force numeric columns to avoid type inference errors with mixed content
    numeric_cols = ['year', 'parameters', 'hardware_compute', 'hardware_number', 
                    'hardware_power', 'training_compute', 'training_time', 
                    'power_draw', 'co2eq']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.where(pd.notnull(df), None)
    table = db.create_table('greenmir', primary_id='id', primary_type=db.types.integer)
    table.insert_many(df.to_dict(orient='records'))

    paper_info_table = create_paper_info_table(db)
    model_info_table = create_model_info_table(db)

    for id_paper, row in df.iterrows():


        paper_info_table.insert({
            "link": row['link'],
            "abstract": row['abstract'],
            "country": row['country'],
            "id_country": get_country_id(row['country'], db),
            "year": row['year'],

        })

        id_paper+=1

        model_info_table.insert({
            "id_paper": id_paper,
            "model": row['model name'],
            
            "parameters": row['parameters'],
            "hardware": row['hardware'],
            "id_hardware": get_hardware_id(row['hardware'], db),

            "hardware_compute": row['hardware_compute'],
            "hardware_number": row['hardware_number'],
            "hardware_power": row['hardware_power'],

            "training_compute": row['training_compute'],
            "training_time": row['training_time'],
            "power_draw": row['power_draw'],

            "co2eq": row['co2eq'],
        })

    create_paper_document_table(db)
    create_paper_text_table(db)

if __name__ == '__main__':
    if os.path.exists('.cache/tests/greenmir.db'):
        os.remove('.cache/tests/greenmir.db')
    os.makedirs('.cache/tests', exist_ok=True)
    db = dataset.connect('sqlite:///.cache/tests/greenmir.db')
    create_greenmir_table(db)
