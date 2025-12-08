from urllib.parse import urlparse

import dataset
import requests
from sqlalchemy import LargeBinary

import src.tables.paper_text as paper_text


def create_paper_document_table(db: dataset.Database):
    # Ensure a fresh table to avoid PK collisions on reruns.
    if 'paper_document' in db:
        db['paper_document'].drop()

    table = db.create_table('paper_document', primary_id='id_paper', primary_type=db.types.integer)
    doc_type = next((getattr(db.types, name) for name in ('binary', 'LargeBinary', 'BINARY', 'Blob', 'BLOB') if hasattr(db.types, name)), LargeBinary)
    table.create_column('document', doc_type)

    # Use rows directly from the dataset table (no iterrows on Table).
    for idx, row in enumerate(db['paper_info'], start=1):
        link = row.get('link')
        print(f"Downloading paper {idx} (id={row.get('id_paper')}): {link}")
        doc = get_document(link)
        try:
            table.insert({
                'id_paper': row.get('id_paper'),
                'document': doc,
            })
        except Exception as exc:
            print(f"[paper_document] failed to insert id_paper={row.get('id_paper')}: {exc}")


def get_document(link: str):
    if not link or not isinstance(link, str):
        return None

    link = process_link(link.strip())
    parsed = urlparse(link)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return None

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
        'Accept': 'application/pdf,application/octet-stream;q=0.8,*/*;q=0.5',
    }
    try:
        resp = requests.get(link, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def process_link(link: str):
    if 'arxiv.org/abs/' in link:
        return link.replace('arxiv.org/abs/', 'arxiv.org/pdf/')

    return link
