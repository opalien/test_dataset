
### Preparation

0) Enter the Nix shell :

```bash
nix develop .nix/
```

1) create environment :

```bash
python -m venv .env/
```

2) enter the environment :

```bash
source .env/bin/activate
```

3) update & install libraries :pip install -r requirements.txt

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Run the code

#### Download the dataset and/or make database

1) Epoch.ai :

```bash
python -m src.make_tables --database epoch
```

2) GreenMir :

```bash
python -m src.make_tables --database greenmir
```

3) from files :

```bash
python -m src.tables.make_table_from_folder --folder PATH_TO_FOLDER --output PATH_TO --
