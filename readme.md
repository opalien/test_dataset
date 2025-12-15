
### Preparation

0) Enter the Nix shell if you have Nix, otherwise skip this step:

```bash
nix develop .nix/
```

1) Create the environment:

```bash
python -m venv .env/
```

2) Enter the environment:

```bash
source .env/bin/activate
```

3) Update & install libraries:

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Run the code

#### Download the dataset and/or make database

1) Epoch.ai:

```bash
python -m src.make_tables --database epoch
```

2) GreenMIR:

```bash
python -m src.make_tables --database greenmir
```

3) From files:

```bash
python -m src.tables.make_table_from_folder --folder PATH_TO_FOLDER --output PATH_TO_OUTPUT --csv PATH_TO_CSV
```

#### Infer

1) Gemini:

```bash
python -m src.infer --database data/greenmir.db --output outputs/greenmir_gemini.json --api-key YOUR_API_KEY --questions-dir src/questions/estimated
```

Available question types: `src/questions/strict`, `src/questions/estimated`, `src/questions/derived`

2) Ollama (with SLURM):

```bash
OLLAMA_MODEL=mistral-3:latest srun --gres=gpu:a100-20:1 --partition=besteffort ./launch.sh --database data/greenmir.db --output outputs/greenmir_ollama.json
```

#### Benchmarks

1) GreenMIR & Epoch.ai:

```bash
python -m src.benchmark greenmir --outputs outputs/greenmir --db data/greenmir.db --results outputs/greenmir_gemini.json
python -m src.benchmark epoch --outputs outputs/epoch --db data/epoch.db --results outputs/epoch_gemini.json --arxiv-only
```

2) From files:

```bash
python -m src.benchmark folder --outputs outputs/folder --db data/custom.db --results outputs/custom_results.json
```

#### Plots

1) GreenMIR & Epoch.ai:

```bash
python -m src.plot greenmir --results-dir outputs/greenmir --output-dir plots/greenmir
python -m src.plot epoch --results-dir outputs/epoch --output-dir plots/epoch
```

2) From files:

```bash
python -m src.plot folder --results-dir outputs/folder --output-dir plots/folder
```


### Structure

#### Important folders

The `src` folder contains the entire program. The `static` folder contains the GreenMIR hand-reshaped dataset.

The main programs to run are:

- `src.make_tables`: Download the dataset and create the database (`src.tables` contains all the table definitions)
- `src.infer`: Infer the dataset with Gemini
- `src.infer_ollama`: Infer the dataset with Ollama using SLURM
- `src.benchmark`: Run the benchmarks (`src.benchmarks` contains all the benchmark definitions)
- `src.plot`: Generate the plots

Other programs for file inference:

- `src.tables.make_table_from_folder`: Create a table from a folder of files


#### General structure

```bash
├── data/                                      # Generated databases
│   ├── greenmir.db
│   └── epoch.db
├── outputs/                                   # Generated results
│   └── results.json
├── src/
│   ├── infer.py                               # Infer with Gemini
│   ├── infer_ollama.py                        # Infer with Ollama
│   ├── make_tables.py                         # Create the databases
│   ├── plot.py                                # Plot the results
│   ├── benchmark.py                           # Run the benchmarks
│   ├── questions/                             # Questions for inference
│   │   └── {strict, estimated, derived}/      # Different levels of questions
│   │       ├── co2eq.txt
│   │       ├── country.txt
│   │       ├── h_compute.txt
│   │       ├── h_number.txt
│   │       ├── h_power.txt
│   │       ├── hardware.txt
│   │       ├── model_enumeration.txt
│   │       ├── parameters.txt
│   │       ├── training_compute.txt
│   │       ├── training_time.txt
│   │       └── year.txt
│   ├── benchmarks/                            # Benchmarks
│   │   ├── benchmark_epoch_id.py
│   │   ├── benchmark_folder_id.py
│   │   ├── benchmark_greenmir_id.py
│   │   └── core/                              # Core benchmarks (shared utilities)
│   └── tables/                                # Tables
│       ├── country.py
│       ├── epoch.py
│       ├── greenmir.py
│       ├── hardware.py
│       ├── info_tables.py
│       ├── make_table_from_folder.py          # Create a table from a folder of files
│       ├── paper_document.py
│       └── paper_text.py
├── static/                                    # Static files
│   └── dataset.csv                            # Handmade GreenMIR dataset
├── launch.sh                                  # Launch inference with SLURM and Ollama
├── readme.md
└── requirements.txt
```

