#!/usr/bin/env python3
"""
Script pour créer une base de données SQLite à partir d'un dossier de PDFs.

Usage:
    python -m src.tables.make_table_from_folder --folder /path/to/pdfs --output database.db
    python -m src.tables.make_table_from_folder --folder /path/to/pdfs --output database.db --csv metadata.csv
"""

import argparse
import os
from pathlib import Path

import dataset
import pandas as pd
from sqlalchemy import LargeBinary

from src.tables.country import get_country_id, create_country_table
from src.tables.hardware import get_hardware_id, create_hardware_table
from src.tables.info_tables import create_paper_info_table, create_model_info_table
from src.tables.paper_text import create_paper_text_table


def create_paper_document_table_from_folder(db: dataset.Database, folder: Path):
    if 'paper_document' in db:
        db['paper_document'].drop()

    table = db.create_table('paper_document', primary_id='id_paper', primary_type=db.types.integer)
    doc_type = next(
        (getattr(db.types, name) for name in ('binary', 'LargeBinary', 'BINARY', 'Blob', 'BLOB') if hasattr(db.types, name)),
        LargeBinary
    )
    table.create_column('document', doc_type)
    table.create_column('filename', db.types.string)

    pdf_files = sorted(folder.glob('*.pdf'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    print(f"Found {len(pdf_files)} PDF files in {folder}")

    for pdf_path in pdf_files:
        # id_paper est le nom du fichier sans extension (0.pdf -> 0, 1.pdf -> 1, etc.)
        id_paper = int(pdf_path.stem) if pdf_path.stem.isdigit() else None
        if id_paper is None:
            print(f"[paper_document] skipping {pdf_path.name}: filename is not a valid id_paper")
            continue
        
        print(f"Loading paper id_paper={id_paper}: {pdf_path.name}")
        try:
            doc = pdf_path.read_bytes()
            table.insert({
                'id_paper': id_paper,
                'document': doc,
                'filename': pdf_path.name,
            })
        except Exception as exc:
            print(f"[paper_document] failed to load {pdf_path.name}: {exc}")


def load_csv_metadata(db: dataset.Database, csv_path: Path):
    df = pd.read_csv(csv_path)
    
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    
    numeric_cols = ['id_paper', 'year', 'parameters', 'hardware_compute', 'hardware_number', 
                    'hardware_power', 'training_compute', 'training_time', 
                    'power_draw', 'co2eq']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.where(pd.notnull(df), None)
    
    paper_info_table = create_paper_info_table(db)
    model_info_table = create_model_info_table(db)
    
    for idx, row in df.iterrows():
        id_paper = int(row['id_paper']) if pd.notna(row.get('id_paper')) else None
        if id_paper is None:
            print(f"Warning: row {idx} has no id_paper, skipping...")
            continue
        
        country = row.get('country')
        hardware = row.get('hardware')
        
        if paper_info_table.find_one(id_paper=id_paper) is None:
            paper_info_table.insert({
                'id_paper': id_paper,
                'link': row.get('link'),
                'abstract': row.get('abstract'),
                'country': country,
                'id_country': get_country_id(country, db) if country else None,
                'year': row.get('year'),
                'split': row.get('split'),
            })
        
        model_name = row.get('model') or row.get('model_name')
        if model_name:
            model_info_table.insert({
                'id_paper': id_paper,
                'model': model_name,
                'architecture': row.get('architecture'),
                'parameters': row.get('parameters'),
                'hardware': hardware,
                'id_hardware': get_hardware_id(hardware, db) if hardware else None,
                'hardware_compute': row.get('hardware_compute'),
                'hardware_number': row.get('hardware_number'),
                'hardware_power': row.get('hardware_power'),
                'training_compute': row.get('training_compute'),
                'training_time': row.get('training_time'),
                'power_draw': row.get('power_draw'),
                'co2eq': row.get('co2eq'),
            })


def create_folder_table(db: dataset.Database, folder: Path, csv_path: Path = None):
    create_country_table(db)
    create_hardware_table(db)
    
    create_paper_document_table_from_folder(db, folder)
    create_paper_text_table(db)
    
    if csv_path:
        load_csv_metadata(db, csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Crée une base de données SQLite à partir d'un dossier de PDFs."
    )
    parser.add_argument('--folder', '-f', type=str, required=True,
                        help="Chemin vers le dossier contenant les fichiers PDF.")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Chemin du fichier de sortie .db (SQLite).")
    parser.add_argument('--csv', '-c', type=str, default=None,
                        help="Chemin optionnel vers un fichier CSV contenant les métadonnées.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Le dossier {folder} n'existe pas ou n'est pas un répertoire.")

    output_path = args.output
    if not output_path.endswith('.db'):
        output_path += '.db'

    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    db = dataset.connect(f'sqlite:///{output_path}')
    
    csv_path = Path(args.csv) if args.csv else None
    if csv_path and not csv_path.exists():
        raise ValueError(f"Le fichier CSV {csv_path} n'existe pas.")
    
    create_folder_table(db, folder, csv_path)
    
    print(f"\nDatabase created: {output_path}")
    print(f"Tables: {db.tables}")
    for table_name in db.tables:
        print(f"  - {table_name}: {db[table_name].count()} rows")
