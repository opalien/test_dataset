import os
import dataset
import argparse
from src.tables.greenmir import create_greenmir_table
from src.tables.epoch import create_epoch_table

parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default='greenmir')
args = parser.parse_args()


if __name__ == '__main__':
    if args.database == 'greenmir':
        db_path = 'sqlite:///data/greenmir.db'
    elif args.database == 'epoch':
        db_path = 'sqlite:///data/epoch.db'
    else:
        raise ValueError(f"Unsupported database: {args.database}")

    if os.path.exists(db_path.replace('sqlite:///', '')):
        os.remove(db_path.replace('sqlite:///', ''))
    os.makedirs('data/', exist_ok=True)

    db = dataset.connect(db_path)
    if args.database == 'greenmir':
        create_greenmir_table(db)   
    elif args.database == 'epoch':
        create_epoch_table(db)  
