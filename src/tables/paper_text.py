import dataset


def pdf_bytes_to_text(data: bytes) -> str:
    """
    Extract text from a PDF byte stream. Returns empty string on failure.
    """
    if not data:
        return ""
    try:
        import pymupdf
        text_chunks: list[str] = []
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            for page in doc:
                text_chunks.append(page.get_text())
        return "\n".join(text_chunks)
    except Exception as exc:
        # Log and return empty string so pipeline continues
        print(f"[paper_text] failed to extract text: {exc}")
        return ""


def create_paper_text_table(db: dataset.Database):
    if 'paper_text' in db:
        db['paper_text'].drop()

    table = db.create_table('paper_text', primary_id='id_paper', primary_type=db.types.integer)
    table.create_column('text', db.types.text)

    for idx, row in enumerate(db['paper_document'], start=1):
        text = pdf_bytes_to_text(row.get('document'))
        table.insert({
            'id_paper': row.get('id_paper'),
            'text': text
        })
