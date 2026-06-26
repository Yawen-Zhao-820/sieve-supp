# SIEVE

## Environment

Tested with Python 3.9.12 and `torch` 1.12.1, `torchvision` 0.13.1, `numpy` 1.26.4, `pandas` 2.3.3,
`scipy` 1.13.1, `transformers` 4.38.1. `transformers` is only needed for the two text datasets
(MultiNLI, CivilComments), where it loads `bert-base-uncased`.

## Data

Set `ROOT_DIR` for each script to the directory holding that dataset. The expected file layout is
defined in `dataset/<name>_dataset.py`.

Vision datasets follow the DaC setup (https://github.com/fhn98/DaC):

- **Waterbirds**: download https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
  and extract into `ROOT_DIR`.
- **CelebA**: the CelebA images and attribute CSVs (https://www.kaggle.com/jessicali9530/celeba-dataset),
  arranged so `ROOT_DIR/public_datasets/CelebA/celebA/` holds `img_align_celeba/`, `list_attr_celeba.csv`,
  and `list_eval_partition.csv`. The class-balanced train/val/test split we use ships as
  `celeba_metadata.csv` in the `metadata/` directory and is loaded automatically.
- **MetaShift**: we use the implementation provided by the DISC repo (https://github.com/Wuyxin/DISC).
  You can download the dataset from
  https://drive.usercontent.google.com/download?id=1WySOxBRkxAUlSokgZrC-0JaWZwcG5UMT&authuser=0
- **Dominoes**: the code for preparing the Dominoes dataset is provided in
  https://github.com/mpagli/Agree-to-Disagree . You can download a saved instance of the dataset from
  https://drive.google.com/drive/folders/1iXOFqxA6IAWTS_MD9xy3SD7FMTGChC2t?usp=sharing

Text datasets follow the JTT setup (https://github.com/anniesch/jtt):

- **MultiNLI**: follow the MultiNLI-with-annotated-negations instructions
  (https://github.com/kohpangwei/group_DRO#multinli-with-annotated-negations), with `metadata_random.csv`
  and the cached BERT features under `glue_data/MNLI/` inside `ROOT_DIR`.
- **CivilComments**: the CivilComments-WILDS data under `ROOT_DIR`.

## Run

Each dataset has one script under `scripts/`. Set `ROOT_DIR` and run it.

| Script | Dataset | Backbone |
|---|---|---|
| `scripts/run_waterbirds.sh` | Waterbirds | ResNet-50 |
| `scripts/run_celeba.sh` | CelebA | ResNet-50 |
| `scripts/run_metashift.sh` | MetaShift | ResNet-50 |
| `scripts/run_dominoes.sh` | Dominoes | ResNet-50 |
| `scripts/run_multinli.sh` | MultiNLI | BERT-base |
| `scripts/run_civilcomments.sh` | CivilComments | BERT-base |

Each run prints the selection and training milestones, the progress bars, and the final test
worst-group and overall accuracy, and writes them to
`results/.../evaluate_results/<dataset>-final-sieve-erm.txt`. For CivilComments it also reports the
16-group (WILDS) worst-group accuracy.

### Sample commands

```bash
ROOT_DIR=/path/to/waterbirds bash scripts/run_waterbirds.sh
ROOT_DIR=/path/to/celeba bash scripts/run_celeba.sh
ROOT_DIR=/path/to/metashift bash scripts/run_metashift.sh
ROOT_DIR=/path/to/dominoes bash scripts/run_dominoes.sh
ROOT_DIR=/path/to/multinli bash scripts/run_multinli.sh
ROOT_DIR=/path/to/civilcomments bash scripts/run_civilcomments.sh
```
