# FracFeedExtractor - LLMs for the fraction of feeding predators

Using machine learning and LLMs to automatically identify predator diet studies in ecological literature and extract key data on predator feeding rates at scale.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-pending-lightgrey?style=flat-square)
![GitHub Issues](https://img.shields.io/github/issues/NovakLabOSU/FracFeedExtractor?style=flat-square)

## Project Description
This project contributes to validating a novel metric of predator-prey interactions to inform ecosystem-based resource management and ecological theory.  It does so by using a global database of predator diet surveys to train large language models for the purpose of identifying additional publications and extracting key data to overcome the limitations that have hindered the empirical validation of the new metric thus far.

## Key Features
- **PDF Classification** - An XGBoost classifier identifies which scientific publications contain useful predator diet survey data, filtering out irrelevant papers before they reach the LLM.
- **Structured Data Extraction** - Automatically parses empty and non-empty stomach counts and key covariates (predator identity, survey location, survey year, and more) from tabular and narrative text.
- **Batch Processing** - Accepts a single PDF or an entire folder of PDFs in one command.
- **Provenance & Uncertainty Reporting** - Every result includes descriptors of classification confidence and extraction provenance.
- **Reproducible Pipeline** - A clean training and evaluation pipeline with PDF preprocessing and model evaluation metrics is fully documented in this repository.

## Get Started

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running locally (minimum 8 GB RAM; 16 GB recommended)
- Pull the required model before running the pipeline:
  ```bash
  ollama pull qwen2.5:7b   # default extraction model (~5 GB)
  ```
  Verify Ollama is running: `ollama list`

### Installation
```bash
git clone https://github.com/NovakLabOSU/FracFeedExtractor.git
cd FracFeedExtractor
pip install -r requirements.txt
```

> **Note:** Tesseract OCR must be installed separately as a system dependency. See the [Contributing Guide](documentation/CONTRIBUTING.md) for platform-specific instructions.

### Quick Start
```bash
# Classify and extract data from a folder of PDFs
python classify_extract.py path/to/pdfs/
```

For full setup details, virtual environment configuration, available CLI flags, and contribution guidelines, see the [Contributing Guide](documentation/CONTRIBUTING.md).

## Pipeline Demo

[IMAGE: Terminal screenshot showing the pipeline running on a folder of PDFs, displaying classifier output and extraction results per file]

*Terminal screenshot showing the pipeline running on a folder of PDFs, displaying classifier output and extraction results per file.*

---

[IMAGE: Side-by-side comparison of a dense academic PDF on the left and the clean extracted JSON output on the right]

*Side-by-side comparison of a dense academic PDF on the left and the clean extracted JSON output on the right.*

## Motivation
Predator-prey interactions are central to ecosystem stability, yet a key parameter that quantifies predator-prey interaction strength (predator feeding rates) is rarely used in practice because the data required to estimate it are difficult to obtain. Our research has shown that the fraction of feeding individuals, defined as the proportion of predators with non-empty stomachs, can be easily obtained from routine predator diet surveys and is analytically linked to a species' metabolic demand, body size, temperature, mortality rate, extinction susceptibility, biological control effectiveness, and population resilience to perturbations. To validate this metric for mainstream resource management and ecological theory, a scalable method is needed to harvest the untapped data that exists in the vast ecological literature.  

The project trained large language models for two tasks: 1) classifying scientific publications as containing useful predator diet survey information, and 2) extracting the numbers of empty- and non-empty stomachs counted and key covariates (predator identity, survey location, survey year, etc.).  By fine-tuning with a large database of hand-annotated publications containing diet surveys conducted across the globe over the last 135 years, the models learned to recognize relevant publications and parse tabular and narrative data into structured fields. The resulting pipeline enables the generation of a comprehensive, covariate-rich database for subsequent analyses and applications.


## Objectives/Deliverables
1. A fully trained, fine-tuned Python implementation of a large language model (or pair of models) that ingests a publication's pdf and returns a classification and/or the extracted data as well as descriptors of the classification and extraction provenance and uncertainty. 
2. A Python pipeline that accepts a single pdf or a folder of pdfs, parses the text of each, queries the model for each, and exports the classification and data extraction results with clear provenance and uncertainty.  
3. A clean, reproducible training and evaluation pipeline (including pdf preprocessing and model evaluation metrics) documented in a GitHub repository. 
4. A technical report detailing model architecture, training procedure, validation results, and guidance for future extensions.

## Data sources
[FracFeed: Global database of the fraction of feeding predators](https://github.com/marknovak/FracFeed_DB)

## Team

| Name              | Role               | GitHub                                         |
| ----------------- | ------------------ | ---------------------------------------------- |
| Mark Novak        | Project Owner/Lead | [@marknovak](https://github.com/marknovak)     |
| Sean Clayton      | Contributor        | [@SeanClay10](https://github.com/SeanClay10)   |
| Zahra Alsulaimawi | Contributor        | [@QuiteRocks](https://github.com/QuiteRocks)   |
| Raymond Cen       | Contributor        | [@raymondcen](https://github.com/raymondcen)   |
| Bradley Rule      | Contributor        | [@bradleyrule](https://github.com/bradleyrule) |

We also thank all previous contributors - see the full list on the [GitHub Contributors page](https://github.com/NovakLabOSU/FracFeedExtractor/graphs/contributors).

---

Found a bug or have a question? [Open an issue on GitHub Issues](https://github.com/NovakLabOSU/FracFeedExtractor/issues).

## Documentation
- [Contributing Guide](documentation/CONTRIBUTING.md)
- [Pipeline Architecture Diagram](documentation/architecture.png)

LICENSE: Pending partner confirmation
