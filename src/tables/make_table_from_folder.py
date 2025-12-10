#!/usr/bin/env python3
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


def create_paper_document_table_from_folder(db: dataset.Database, folder: Path) -> None:
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
        id_paper = int(pdf_path.stem) if pdf_path.stem.isdigit() else None
        if id_paper is None:
            print(f"[paper_document] skipping {pdf_path.name}: filename is not a valid id_paper")
            continue

        print(f"Loading paper id_paper={id_paper}: {pdf_path.name}")
        try:
            doc = pdf_path.read_bytes()
            table.insert({'id_paper': id_paper, 'document': doc, 'filename': pdf_path.name})
        except Exception as exc:
            print(f"[paper_document] failed to load {pdf_path.name}: {exc}")


def load_paper_info_csv(db: dataset.Database, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    df = df.where(pd.notnull(df), None)

    table = create_paper_info_table(db)

    for _, row in df.iterrows():
        id_paper = int(row['id_paper']) if pd.notna(row.get('id_paper')) else None
        if id_paper is None:
            continue
        country = row.get('country')
        table.insert({
            'id_paper': id_paper,
            'link': row.get('link'),
            'abstract': row.get('abstract'),
            'country': country,
            'id_country': get_country_id(country, db) if country else None,
            'year': int(row['year']) if pd.notna(row.get('year')) else None,
            'split': row.get('split'),
        })
    print(f"[paper_info] loaded {table.count()} papers from {csv_path.name}")


def load_model_info_csv(db: dataset.Database, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    numeric_cols = ['id_paper', 'parameters', 'hardware_compute', 'hardware_number',
                    'hardware_power', 'training_compute', 'training_time', 'power_draw', 'co2eq']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.where(pd.notnull(df), None)

    table = create_model_info_table(db)

    for _, row in df.iterrows():
        id_paper = int(row['id_paper']) if pd.notna(row.get('id_paper')) else None
        if id_paper is None:
            continue
        hardware = row.get('hardware')
        model_name = row.get('model') or row.get('model_name')
        table.insert({
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
    print(f"[model_info] loaded {table.count()} models from {csv_path.name}")


def load_combined_csv(db: dataset.Database, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    numeric_cols = ['id_paper', 'year', 'parameters', 'hardware_compute', 'hardware_number',
                    'hardware_power', 'training_compute', 'training_time', 'power_draw', 'co2eq']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.where(pd.notnull(df), None)

    paper_table = create_paper_info_table(db)
    model_table = create_model_info_table(db)
    seen_papers: set[int] = set()

    for _, row in df.iterrows():
        id_paper = int(row['id_paper']) if pd.notna(row.get('id_paper')) else None
        if id_paper is None:
            continue

        if id_paper not in seen_papers:
            country = row.get('country')
            paper_table.insert({
                'id_paper': id_paper,
                'link': row.get('link'),
                'abstract': row.get('abstract'),
                'country': country,
                'id_country': get_country_id(country, db) if country else None,
                'year': int(row['year']) if pd.notna(row.get('year')) else None,
                'split': row.get('split'),
            })
            seen_papers.add(id_paper)

        model_name = row.get('model') or row.get('model_name')
        if model_name:
            hardware = row.get('hardware')
            model_table.insert({
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

    print(f"[combined] loaded {paper_table.count()} papers, {model_table.count()} models from {csv_path.name}")


def create_folder_table(
    db: dataset.Database,
    folder: Path,
    paper_csv: Path | None = None,
    model_csv: Path | None = None,
    combined_csv: Path | None = None
) -> None:
    create_country_table(db)
    create_hardware_table(db)
    create_paper_document_table_from_folder(db, folder)
    create_paper_text_table(db)

    if combined_csv:
        load_combined_csv(db, combined_csv)
    else:
        if paper_csv:
            load_paper_info_csv(db, paper_csv)
        if model_csv:
            load_model_info_csv(db, model_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create SQLite database from a folder of PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.tables.make_table_from_folder -f pdfs/ -o out.db
  python -m src.tables.make_table_from_folder -f pdfs/ -o out.db --paper-csv paper_info.csv --model-csv model_info.csv
  python -m src.tables.make_table_from_folder -f pdfs/ -o out.db --csv combined.csv
        """
    )
    parser.add_argument('--folder', '-f', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--paper-csv', type=str, help="CSV with paper_info columns (id_paper, link, abstract, country, year, split)")
    parser.add_argument('--model-csv', type=str, help="CSV with model_info columns (id_paper, model, parameters, hardware, ...)")
    parser.add_argument('--csv', '-c', type=str, help="Single CSV with both paper and model info (epoch-style: one row per model)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder {folder} does not exist or is not a directory.")

    output_path = args.output if args.output.endswith('.db') else args.output + '.db'
    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    db = dataset.connect(f'sqlite:///{output_path}')

    paper_csv = Path(args.paper_csv) if args.paper_csv else None
    model_csv = Path(args.model_csv) if args.model_csv else None
    combined_csv = Path(args.csv) if args.csv else None

    for p in [paper_csv, model_csv, combined_csv]:
        if p and not p.exists():
            raise ValueError(f"CSV file {p} does not exist.")

    create_folder_table(db, folder, paper_csv, model_csv, combined_csv)

    print(f"\nDatabase created: {output_path}")
    print(f"Tables: {db.tables}")
    for table_name in db.tables:
        print(f"  - {table_name}: {db[table_name].count()} rows")
