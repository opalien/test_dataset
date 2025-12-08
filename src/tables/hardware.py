import dataset
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

HARDWARE_URL = "https://epoch.ai/data/generated/ml_hardware.zip"

def create_hardware_table(db: dataset.Database):
    with urlopen(HARDWARE_URL) as r, ZipFile(BytesIO(r.read())) as z:
        with z.open(next(n for n in z.namelist() if n.endswith('.csv'))) as f:
            df = pd.read_csv(f)

    # Map potential CSV columns to target schema
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {
        'name': ['hardware name', 'hardware', 'name'],
        'compute': ['compute', 'fp32 (single precision) performance (flop/s)'],
        'power': ['power', 'tdp_w', 'tdp (w)']
    }
    
    src = {k: next((cols[c] for c in v if c in cols), None) for k, v in mapping.items()}

    out = pd.DataFrame({
        'name': df[src['name']].astype(str) if src['name'] else None,
        'compute': pd.to_numeric(df[src['compute']], errors='coerce') if src['compute'] else None,
        'power': pd.to_numeric(df[src['power']], errors='coerce') / 1000.0 if src['power'] else None
    }).where(lambda x: x.notnull(), None)

    db['hardware'].insert_many(out.to_dict(orient='records'))


def get_hardware_id(hardware_name: str, db: dataset.Database):
    if not hardware_name:
        return None
    hardware_name = hardware_name.strip()

    if 'hardware' not in db.tables:
        create_hardware_table(db)

    table = db['hardware']
    if table.count() == 0:
        create_hardware_table(db)

    hardware_list = list(table)
    if not hardware_list:
        return None

    best_match = min(hardware_list, key=lambda row: JaroWinkler.normalized_distance(hardware_name, row['name']))
    return best_match['id']
    
if __name__ == '__main__':
    import os
    if os.path.exists('.cache/tests/hardware.db'):
        os.remove('.cache/tests/hardware.db')
    os.makedirs('.cache/tests', exist_ok=True)
    create_hardware_table(dataset.connect('sqlite:///.cache/tests/hardware.db'))
