import dataset

def create_paper_info_table(db: dataset.Database):
    paper_info_table = db.create_table('paper_info', primary_id='id_paper', primary_type=db.types.integer)
    paper_info_table.create_column('link', db.types.string)
    paper_info_table.create_column('abstract', db.types.string)
    paper_info_table.create_column('country', db.types.string)
    paper_info_table.create_column('id_country', db.types.integer)
    paper_info_table.create_column('year', db.types.integer)
    paper_info_table.create_column('split', db.types.string)
    return paper_info_table

def create_model_info_table(db: dataset.Database):
    model_info_table = db.create_table('model_info', primary_id='id_model', primary_type=db.types.integer)
    model_info_table.create_column('id_paper', db.types.integer)
    model_info_table.create_column('model', db.types.string)
    model_info_table.create_column('architecture', db.types.string)
    model_info_table.create_column('parameters', db.types.integer)
    model_info_table.create_column('id_hardware', db.types.integer)
    model_info_table.create_column('hardware', db.types.string)
    model_info_table.create_column('hardware_compute', db.types.integer)
    model_info_table.create_column('hardware_number', db.types.integer)
    model_info_table.create_column('hardware_power', db.types.integer)
    model_info_table.create_column('training_compute', db.types.integer)
    model_info_table.create_column('training_time', db.types.integer)
    model_info_table.create_column('power_draw', db.types.integer)
    model_info_table.create_column('co2eq', db.types.integer)
    return model_info_table