# FracFeedExtractor - LLMs for the fraction of feeding predators

Using machine learning and LLMs to automatically identify predator diet studies in ecological literature and extract key data on predator feeding rates at scale.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-pending-lightgrey?style=flat-square)
![GitHub Issues](https://img.shields.io/github/issues/NovakLabOSU/FracFeedExtractor?style=flat-square)

## Project Description
This project contributes to validating a novel metric of predator-prey interactions to inform ecosystem-based resource management and ecological theory. It does so by using a global database of predator diet surveys to train an XGBoost classifier that identifies relevant publications and a pre-trained LLM running locally via Ollama that extracts key data to overcome the limitations that have hindered the empirical validation of the new metric thus far.

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

The project trained an XGBoost classifier to identify which publications contain useful predator diet survey information. A pre-trained LLM running locally via Ollama then extracts the numbers of empty and non-empty stomachs counted and key covariates (predator identity, survey location, survey year, etc.).  The classifier was trained on a large database of hand-annotated publications containing diet surveys conducted across the globe over the last 135 years, learning to recognize relevant publications so the LLM only processes papers likely to contain useful data. The resulting pipeline enables the generation of a comprehensive, covariate-rich database for subsequent analyses and applications.


## Objectives/Deliverables
1. A trained XGBoost classifier paired with a pre-trained LLM that together ingest a publication's PDF and return a classification and the extracted data with descriptors of classification confidence and extraction provenance. 
2. A Python pipeline that accepts a single pdf or a folder of pdfs, parses the text of each, queries the model for each, and exports the classification and data extraction results with clear provenance and uncertainty.  
3. A clean, reproducible training and evaluation pipeline (including pdf preprocessing and model evaluation metrics) documented in a GitHub repository. 
4. A documented GitHub repository detailing the classifier architecture, training procedure and guidance for future extensions.

## Data sources
[FracFeed: Global database of the fraction of feeding predators](https://github.com/marknovak/FracFeed_DB)

## Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/marknovak">
        <img src="https://github.com/marknovak.png" width="80px" alt="Mark Novak"/><br/>
        <b>Mark Novak</b><br/>
        <sub>Project Owner/Lead</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SeanClay10">
        <img src="https://github.com/SeanClay10.png" width="80px" alt="Sean Clayton"/><br/>
        <b>Sean Clayton</b><br/>
        <sub>Contributor</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/QuiteRocks">
        <img src="https://github.com/QuiteRocks.png" width="80px" alt="Zahra Alsulaimawi"/><br/>
        <b>Zahra Alsulaimawi</b><br/>
        <sub>Contributor</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/raymondcen">
        <img src="https://github.com/raymondcen.png" width="80px" alt="Raymond Cen"/><br/>
        <b>Raymond Cen</b><br/>
        <sub>Contributor</sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/bradleyrule">
        <img src="https://github.com/bradleyrule.png" width="80px" alt="Bradley Rule"/><br/>
        <b>Bradley Rule</b><br/>
        <sub>Contributor</sub>
      </a>
    </td>
  </tr>
</table>


---

Found a bug or have a question? [Open an issue on GitHub Issues](https://github.com/NovakLabOSU/FracFeedExtractor/issues).

## Documentation
- [Contributing Guide](documentation/CONTRIBUTING.md)
- [Pipeline Architecture Diagram](documentation/architecture.png)

LICENSE: Pending partner confirmation
