# Contributing Guide
How to set up, code, test, review, and release so contributions meet our Definition
of Done.
## Code of Conduct
All contributors must follow the Oregon State University Student Code of Conduct and the team’s charter agreement.
* Treat all collaborators with respect and professionalism.
* Provide decent participation during meetings and reviews.
* Raise the issue privately with the team first.
* Issues of academic or ethical concern should be reported directly to the instructor.
* Report any inappropriate or unprofessional behavior to the TA, instructor or project manager.
  
**Owner**: Bradley Rule 
**Next Review**:  05/15/26

## Getting Started
> **Pipeline diagram**: [`documentation/architecture.png`](architecture.png) - visual overview of the full extraction pipeline.

* ### Prerequisites
  * Python 3.10 +
  * pip installed
  * Access to GitHub repository
  * [Ollama](https://ollama.com) installed and running locally
    * Minimum hardware: 8 GB RAM (16 GB recommended for `llama3.1:8b`)
    * Pull the required models before running the classify/extract pipeline:
      ```bash
      ollama pull llama3.1:8b   # default extraction model (~5 GB)
      ollama pull qwen2.5:7b    # alternative model (~5 GB)
      ```
    * Verify Ollama is running: `ollama list`
* ### Setup Instructions
```
    git clone https://github.com/NovakLabOSU/FracFeedExtractor.git
    cd FracFeedExtractor
    python -m venv venv
    source venv/bin/activate   
    # Windows: venv\Scripts\activate
    pip install -r requirements.txt
```
* ### Running the application
  * If you do have the [dataset](https://drive.google.com/drive/u/2/folders/1U3_-TmnXnuBPR9vukkyV-3ITsxPr-nfo) downloaded locally on your machine:
    ```
    python scripts/full_pipeline.py --local <file_path_to_dataset>
    ```
    * Note: Your local dataset folder should be formatted like so:
      ```
      folder
        |-> useful
            |-> pdfs
            |-> ...
        |-> not-useful
            |-> pdfs
            |-> ...
      ```
  * If you do not have the dataset downloaded or cannot have it on your locally on your machine:
    ```
    python scripts/full_pipeline.py --api
    ```
    * Note: You will need access to the .env file
* ### Running the classify/extract pipeline
  Use `classify_extract.py` to classify PDFs and extract structured diet data in a single step.
  Requires trained model artifacts in `src/model/models/` (run the full pipeline first,
  or see [Retraining the Classifier](#retraining-the-classifier-and-extending-extraction) below).
  ```bash
  # Single PDF
  python classify_extract.py path/to/file.pdf

  # Folder of PDFs (sequential)
  python classify_extract.py path/to/pdfs/

  # All options
  python classify_extract.py path/to/pdfs/ \
      --model-dir src/model/models \
      --llm-model llama3.1:8b \
      --output-dir data/results \
      --confidence-threshold 0.70 \
      --max-chars 12000 \
      --num-ctx 4096 \
      --workers 4
  ```
  | Flag | Default | Description |
  |------|---------|-------------|
  | `--model-dir` | `src/model/models` | Directory containing classifier artifacts |
  | `--llm-model` | `llama3.1:8b` | Ollama model for extraction |
  | `--output-dir` | `data/results` | Destination for JSON results and summary CSV |
  | `--confidence-threshold` | `0.70` | Probability threshold for "useful" classification |
  | `--max-chars` | `12000` | Maximum characters sent to the LLM |
  | `--num-ctx` | `4096` | Ollama context window size (tokens) |
  | `--workers` | `1` | Parallel worker processes (`1` = sequential) |

* ### Sample Output
  Each PDF classified as "useful" produces a JSON file in `data/results/metrics/`:
  ```json
  {
    "source_file": "Smith_2002.pdf",
    "extracted_at": "2026-04-24T14:32:00",
    "metrics": {
      "species_name": "Esox lucius",
      "study_location": "Lake Windermere, UK",
      "study_date": "1998-2000",
      "num_empty_stomachs": 42,
      "num_nonempty_stomachs": 158,
      "sample_size": 200,
      "fraction_feeding": 0.79
    }
  }
  ```
  A timestamped summary CSV is written to `data/results/summaries/pipeline_summary_<timestamp>.csv`
  with one row per PDF:

  | filename | classification | confidence | pred_prob | extraction_status | species_name | study_location | study_date | sample_size | num_empty_stomachs | num_nonempty_stomachs | fraction_feeding |
  |----------|----------------|------------|-----------|-------------------|--------------|----------------|------------|-------------|--------------------|-----------------------|------------------|
  | Smith_2002.pdf | useful | 0.9231 | 0.9231 | success | Esox lucius | Lake Windermere, UK | 1998-2000 | 200 | 42 | 158 | 0.79 |
  | Jones_1999.pdf | not useful | 0.1204 | 0.1204 | skipped_not_useful | | | | | | | |

* ### Environment Variables
    * Sensitive information such as API keys will be stored in a local .env file which will be excluded by .gitignore.
    * Never hardcode secrets
  
**Owner**: Raymond Cen
**Next Review**:  05/15/26

## Branching & Workflow
We will use the feature-branch workflow with all merges handled through PRs.
* Default branch: ```main```
* Branch naming template:
  * ```feature/short-name```
  * ```bugfix/short-name```
* Rebase your working branch with main, and often, before submitting a PR (simpler conflict resolution)
  
**Owner**: Zahra Zahir Ahmed Alsulaimawi
**Next Review**:  05/15/26

## Issues & Planning
Issue titles should start with the following tags to designate intent:
- `FEAT:` New feature request.
  - Include problem and feature description within issue description field.
  - Include note of requirement ID where applicable (ex: `Requirement ID: REQ-005`).
- `BUG:`  Bug report.
  - Include problem description.
- `DOC:`  Documentation changes.
  - Brief description of what needs to be changed or added.

Example:
```
FEAT: Example Issue

Requirement ID: REQ-XXX

Problem Desciption:
- Brief explanation of why this is important

Feature Description:
- Brief explanation of expected implementation 
```

**Owner**: Zahra Zahir Ahmed Alsulaimawi
**Next Review**:  05/15/26

## Commit Messages
We will use the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) format for clarity and traceability

```
<type>(scope): short summary [issue number if applicable]

[optional body]
[optional footer]
```
**Examples:**
```
feat(parser): implement CSV input parsing 
fix(ci): update pytest command in workflow [#42]
docs(readme): add setup section
```
**Owner**: Raymond Cen
**Next Review**:  05/15/26

## Code Style, Linting & Formatting
We use Black for automatic code formatting and Flake8 for linting to maintain consistent style and prevent common Python errors.

* ### Formatter: Black
  - Config file: `pyproject.toml`
  - Install `pip install black`
  - Local usage:
  ```bash
  # Check formatting without changing files
  black --check src tests

  # Automatically reformat code
  black src tests
  ```

* ### Linter: Flake8
  - Config file: `pyproject.toml`
  - Install `pip install flake8`
  - Local usage:
  ```bash
  flake8 src/ tests/
  ```
  - Configured to ignore line length violations (E501) and other minor style differences.

**Owner**: Sean Clayton
**Next Review**:  05/15/26


## Testing
* ### Test framework
  - `pytest`

* ### Running tests locally
  ```bash
  # Run all tests
  pytest tests/

  # Run tests with coverage
  coverage run -m pytest tests/
  coverage report -m
  coverage html
  ```
**Owner**: Sean Clayton
**Next Review**:  05/15/26

* ### Expectations
    - New features must include unit or integration tests.
    - Coverage thresholds: aim for >80% for core modules; all critical paths must be tested.
    - Tests must pass locally before creating a PR.

## Pull Requests & Reviews
* ### PR requirements
  - Keep PRs focused and small (**<300 lines changed** if possible).
  - Include related issue references in the PR description.
  - Clearly describe what the PR changes and why.

* ### Review process
  - At least one approving review is required for non-trivial changes.
  - Reviewers check code quality, tests, CI status, and adherence to style guides.
  - Ensure all linting and formatting checks pass.

* ### Approval rules
  - CI must pass all mandatory jobs before merging.
  - PRs should be rebased on the latest `main` branch before merge if there are conflicts.

**Owner**: Bradley Rule
**Next Review**:  05/15/26

## CI/CD
Continuous integration ensures all contributions meet quality standards automatically.

* ### Pipeline
  - GitHub Actions workflow: `.github/workflows/pdf_extraction_ci.yml`

* ### Mandatory jobs
  - Install dependencies
  - Code style checks: Black & Flake8
  - Unit tests & coverage
  - Functional validation: PDF extraction tests

* ### Viewing logs
  - Navigate to the **Actions** tab in GitHub.
  - Select the workflow run and expand individual jobs to see logs.

* ### Before merge
  - All jobs must complete successfully.
  - Any failing test, linter, or formatting check blocks the merge.
  - Artifacts (e.g., coverage reports) are uploaded automatically and can be reviewed.

**Owner**: Sean Clayton
**Next Review**:  05/15/26

## Security & Secrets
State how to report vulnerabilities, prohibited patterns (hard-coded secrets),
dependency update policy, and scanning tools.

* Never commit sensitive credentials, tokens, or API keys.
* Secrets are stored locally in .env files and excluded via .gitignore.
* Dependencies are managed in requirements.txt.
* Use pip-audit monthly to check for vulnerabilities.
* Security issues or potential breaches should be reported privately to the Project Manager and TA.

**Owner**: Raymond Cen
**Next Review**:  05/15/26

## Documentation Expectations

- `README.md`: Provides an overview of the project and its goals. It should also detail where to find information and contact individuals.
- `documentation/`: This directory will store all documentation for the program and its features including standards and usage information. 
- `Comments`: 
  - General comments should be kept short and to the point.
  - Functions should contain a docstring as the first statement within a function providing a brief explanation of the function and a quick explanation of parameters and return values.
  - Inline comments should be reserved for places where the function of code is difficult to understand or infer.

**Owner**: Zahra Zahir Ahmed Alsulaimawi
**Next Review**:  05/15/26

## Release Process
### Versioning Scheme
- We follow **semantic versioning****: `MAJOR.MINOR.PATCH`
  - **MAJOR**: breaking changes
  - **MINOR**: new features, backwards-compatible
  - **PATCH**: bug fixes, minor improvements
- Example: `1.2.0` → `1.2.1` (patch), `1.3.0` (minor), `2.0.0` (major)

### Tagging
- Each release must be tagged in Git using the version number:

```bash
git tag -a v1.2.0 -m "Release v1.2.0: <short description>"
git push origin v1.2.0
```

### Changelog Generation
* For each release, include:

  * Added: new features
  * Changed: updates or improvements
  * Fixed: bug fixes

Example entry:
```
## [1.2.0] - 2025-11-10
### Added
- New feed extraction method
### Fixed
- Timeout handling in API fetch
```

### Packaging and Publishing
* Before publishing, ensure all CI/CD pipelines pass.
* Prepare the release branch or merge main into it.
* Include updated documentation and changelog.

### Rollback Process
* In case of a faulty release
  1) Revert the release tag in Git:
    ```
    git tag -d v1.2.0
    git push origin :refs/tags/v1.2.0
    ```
  2) Revert the merge commit on the main branch:
    ```
    git revert -m 1 <merge-commit-hash>
    git push origin main
    ```
  3) Update CHANGELOG to reflect the rollback.
  4) Notify the team and project partner of the rollback.

**Owner**: Bradley Rule 
**Next Review**:  05/15/26

## Retraining the Classifier and Extending Extraction

### Retraining the XGBoost Classifier

The classifier artifacts are saved in `src/model/models/`. To retrain with new or updated labeled data:

1. **Add labeled text files** to `data/processed-text/` and update `data/labels.json`
   with `"filename.txt": "useful"` or `"filename.txt": "not useful"` entries.

2. **Run the trainer directly:**
   ```bash
   python src/model/train_model.py
   ```
   This reads from `data/processed-text/` and `data/labels.json`, trains a TF-IDF +
   XGBoost model, and saves three artifacts:
   - `src/model/models/pdf_classifier.json` - XGBoost model
   - `src/model/models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
   - `src/model/models/label_encoder.pkl` - LabelEncoder

3. **Or run the full pipeline**, which trains the model as a final step:
   ```bash
   python scripts/full_pipeline.py --local <path_to_dataset>
   ```

Key tunable parameters in `src/model/train_model.py`:
- `max_features` in `TfidfVectorizer` (default: 10,000)
- `eta`, `max_depth`, `subsample` in the XGBoost `params` dict
- `early_stopping_rounds` (default: 20)

### Adding New Extraction Fields to the LLM Extractor

Extraction fields are defined in two places:

1. **`src/llm/models.py`** - the `PredatorDietMetrics` Pydantic model.
   Add a new optional field with the appropriate type and a `None` default:
   ```python
   prey_taxa: Optional[list[str]] = None
   ```

2. **`src/llm/llm_client.py`** - the system prompt that instructs the LLM.
   Add a description of the new field and its expected format to the prompt string.

3. **`classify_extract.py`** and **`extract-from-txt.py`** - update the `row` dict
   and `fieldnames` list in the summary CSV writer to include the new column.

After adding a field, run `pytest tests/test_llm_text.py` to verify that the prompt
changes do not break existing extraction tests.

## Support & Contact
* **Primary Communications**: Slack and Teams
* **Meetings**: Fridays 1 PM PST
* **Project Partner**: Mark Novak, Fridays 8:30AM PST (biweekly check-ins) 
* **TA Meetings**: Thursdays 1:30PM PST

**Owner**: Zahra Zahir Ahmed Alsulaimawi
**Next Review**:  05/15/26