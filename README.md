# Titanic Disaster – Survival Prediction (Python & R in Docker)

This repo predicts which passengers survived the sinking of the RMS Titanic (April 15, 1912) using
**logistic regression**. It does the same task twice, each in its own Docker container:

| Container | Language | Code | What it does |
|-----------|----------|------|--------------|
| `titanic-python` | Python 3.11 (pandas, scikit-learn) | `src/python_app/` | Loads data → cleans & engineers features → fits logistic regression → reports accuracy → predicts the test set |
| `titanic-r` | R 4.4.1 (dplyr, base `glm`) | `src/r_app/` | Same pipeline in R, with a slightly different feature set |

Both scripts print every step to the terminal (data loaded, every value filled in, every feature
added, model coefficients, train and validation accuracy, test predictions), so **you only need to
read the terminal output; you don't need to open the code.**

> **The dataset is not included in this repo** (Kaggle's rules, and data should not be committed).
> Step 2 below shows how to download it.

---

## Quick start (TL;DR)

```bash
git clone https://github.com/ohuhohou/titanic-disaster.git
cd titanic-disaster
# put train.csv and test.csv from Kaggle into src/data/   (see Step 2)

docker build -t titanic-python -f src/python_app/Dockerfile .
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python

docker build -t titanic-r -f src/r_app/Dockerfile .
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-r
```

Every command is run from the **repository root** (the `titanic-disaster` folder). The detailed steps follow.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS              # only the repo owner can approve/push changes
├── .gitignore                  # ignores src/data/*, *.csv, .DS_Store, ._*, virtual envs, R/Python caches
├── README.md
└── src/
    ├── data/                   # <- put train.csv and test.csv here (git-ignored; .gitkeep keeps the folder)
    │   └── .gitkeep
    ├── python_app/
    │   ├── Dockerfile          # python:3.11-slim, installs requirements.txt, runs the script
    │   ├── requirements.txt    # pinned package versions
    │   └── titanic_model.py
    └── r_app/
        ├── Dockerfile          # rocker/r-ver:4.4.1, runs install_packages.R, then the script
        ├── install_packages.R  # installs dplyr (CRAN snapshot pinned by the rocker image)
        └── titanic_model.R
```

---

## Prerequisites

| Tool | Why | Check it works |
|------|-----|----------------|
| [Git](https://git-scm.com/downloads) (or GitHub Desktop) | clone the repo | `git --version` |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | build and run the containers | `docker --version`, and Docker Desktop must be **running** |
| A free [Kaggle](https://www.kaggle.com/) account | download the data | – |

You do **not** need Python or R installed on your computer. Docker provides them.

---

## Step 1 – Clone the repository

```bash
git clone https://github.com/ohuhohou/titanic-disaster.git
cd titanic-disaster
```

(GitHub Desktop: *File → Clone repository → URL*, then open a terminal in that folder.)

---

## Step 2 – Download the data into `src/data/`

The data comes from the Kaggle competition **[Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)**.

### Option A – Kaggle website (easiest)

1. Sign in to Kaggle and open <https://www.kaggle.com/competitions/titanic/data>.
2. If asked, click **"Join Competition"** / accept the rules. Kaggle won't let you download until you do.
3. Click **"Download All"**. You get `titanic.zip`.
4. Unzip it and move **`train.csv`** and **`test.csv`** into this repo's `src/data/` folder.
   (`gender_submission.csv` is not needed.)

### Option B – Kaggle command line

```bash
pip install kaggle                     # needs an API token in ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token)
kaggle competitions download -c titanic -p src/data
unzip -o src/data/titanic.zip -d src/data && rm src/data/titanic.zip
```

### Check

```bash
ls src/data
# .gitkeep  gender_submission.csv (optional)  test.csv  train.csv
```

The file names must be exactly `train.csv` and `test.csv`.

---

## Step 3 – Run the Python container

```bash
# 1) Build the image (first build takes ~1 minute; later builds are cached)
docker build -t titanic-python -f src/python_app/Dockerfile .

# 2) Run it, mounting your local data folder into the container
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python
```

- `-f src/python_app/Dockerfile .` picks the Python Dockerfile and uses the repo root as the build context.
- `-v "$(pwd)/src/data:/app/src/data"` shares `src/data` with the container. The data is never copied
  into the image, and the predictions file it writes ends up back in your `src/data`.
- `--rm` deletes the container when it finishes.

**Output:** a printed walkthrough labelled by assignment question (Q13–Q18), plus
`src/data/predictions_python.csv` (`PassengerId,Survived`).

---

## Step 4 – Run the R container

```bash
# 1) Build the image (first build takes a few minutes while R packages install; later builds are cached)
docker build -t titanic-r -f src/r_app/Dockerfile .

# 2) Run it
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-r
```

**Output:** the same Q13–Q18 walkthrough from R, plus `src/data/predictions_r.csv`.

---

## Windows users

`$(pwd)` works in **Git Bash** and **PowerShell**. In PowerShell you can also use `${PWD}`:

```powershell
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-python
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-r
```

In the classic **Command Prompt (cmd.exe)**, use `%cd%`:

```cmd
docker run --rm -v "%cd%\src\data:/app/src/data" titanic-python
docker run --rm -v "%cd%\src\data:/app/src/data" titanic-r
```

---

## What the scripts do

Both scripts follow the same steps, and each printed section is labelled with its assignment question:

| Step | Details |
|------|---------|
| **Q13 – Load** | Reads `train.csv` (891 rows). Prints the first rows and survival rates overall, by sex and by class. |
| **Q14 – Clean / engineer** | Extracts **Title** from `Name` (Mr, Mrs, Miss, Master, Rare). Fills missing **Age** with the median age for each title, missing **Fare** with the median and missing **Embarked** with the most common port. Adds **HasCabin** (Cabin is 77% missing), **FamilySize** = SibSp + Parch + 1, **IsAlone**, and **log(1+Fare)**. Encodes the categorical columns and drops ID and free-text columns. All fill-in values are learned from the **training data only** and then applied to the test data, so no test information leaks into the model. |
| **Q15 – Model** | Python: `StandardScaler → LogisticRegression` (13 features). R: `glm(..., family = binomial)` using Title in place of Sex, because Sex and Title together make unregularised glm unstable. |
| **Q16 – Train accuracy** | Accuracy on the full training set, plus accuracy on a stratified 80/20 hold-out split (seed 42) as an honest out-of-sample estimate. |
| **Q17 – Predict test** | Loads `test.csv` (418 rows), applies the same cleaning and writes the predictions CSV. |
| **Q18 – Test accuracy** | See the note below. |

Expected results (the hold-out split uses a fixed seed, so you should get the same numbers):

| | Train accuracy | Hold-out accuracy | Predicted survivors in test |
|--|--|--|--|
| Python | 0.8305 | 0.8380 | 172 / 418 |
| R | 0.8339 | 0.8315 | 170 / 418 |

### Note on test-set accuracy (Q18)

Kaggle's `test.csv` **does not include the `Survived` column**. Kaggle keeps those labels hidden to
score competition entries, so the true test accuracy cannot be computed locally. The scripts handle this as follows:

1. They report the **hold-out validation accuracy** as the best local estimate of test accuracy.
2. They save a Kaggle-ready predictions file. Upload `src/data/predictions_python.csv` or
   `src/data/predictions_r.csv` at <https://www.kaggle.com/competitions/titanic/submit> to get the official test accuracy.
3. If you have a labelled file, save it as `src/data/test_labels.csv` with columns `PassengerId,Survived`.
   The scripts detect it automatically and print the real test accuracy.

---

## Optional – run without Docker

```bash
# Python (3.11 recommended)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r src/python_app/requirements.txt
python src/python_app/titanic_model.py

# R (4.4)
Rscript src/r_app/install_packages.R
Rscript src/r_app/titanic_model.R
```

Run these from the repo root. `.venv/` is already in `.gitignore`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: train.csv not found at /app/src/data/train.csv` | The data isn't in `src/data/`, or you didn't run `docker run` from the repo root, so the `-v` path is wrong. Run `ls src/data` and repeat Step 2. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it is running. |
| `unable to prepare context` / `Dockerfile not found` | Run `docker build` from the repo root. Don't forget the final `.`. |
| Apple Silicon (M1/M2/M3) image or platform errors | Add `--platform linux/amd64` to both `docker build` and `docker run`. |
| Windows: mounted folder is empty in the container | In Docker Desktop → Settings → Resources → File sharing, allow the drive. Use the PowerShell or cmd commands above. |

---

## Branching workflow used

`feature/titanic-model` (code written here) → pull request → `develop` → pull request → `main`.
