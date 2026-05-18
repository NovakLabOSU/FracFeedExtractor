# Design Decisions & Development History

A record of the major technical choices made during the 2025–2026 OSU Senior Capstone project, written for the next team inheriting this codebase.

---

## Timeline Overview

| Period | Milestone |
|---|---|
| Oct 2025 | Repository setup, PDF text extraction, CI scaffolding |
| Nov 2025 | Initial classifier (Logistic Regression → XGBoost), Google Drive API |
| Jan 2026 | Initial LLM integration, table extraction |
| Feb 2026 | LLM module refactor, BioMistral experiment, OCR improvements, full pipeline |
| Mar 2026 | xgboost rework, section filtering, switched to qwen2.5:7b, parallel workers |
| Apr 2026 | LLM retry logic, Pydantic fixes, classifier loading bug fix, test coverage |
| May 2026 | Repository restructure, handoff cleanup |

---

## 1. Classifier: Logistic Regression → XGBoost

**What we started with:** The first classifier (PR #14, Nov 2025) was a straightforward TF-IDF + Logistic Regression model. It was fast to prototype and gave us a working skeleton for the CI pipeline early on.

**Why we switched:** Logistic Regression's linear decision boundary wasn't capturing the vocabulary patterns that distinguish useful predator diet papers from unrelated ecology papers. After training on the [FracFeed global database](https://github.com/marknovak/FracFeed_DB), XGBoost consistently outperformed it by a meaningful margin.

**What we added around the switch:** When moving to XGBoost (PR #20), we added L2 regularization and explicit class balancing, since the labeled dataset wasn't perfectly balanced between useful and not-useful papers. Both of these had a measurable effect on recall for the useful class specifically. We also enabled GPU training when available, though the CPU path works fine.

**The current model:** TF-IDF vectorizer (10,000 features) feeding an XGBoost booster, trained with early stopping at 20 rounds. Artifacts saved to `src/classifier/models/`. The `pdf_classifier.json` file is the XGBoost model; `tfidf_vectorizer.pkl` and `label_encoder.pkl` are the sklearn preprocessing objects.

---

## 2. LLM Choice: Generic Ollama → BioMistral → qwen2.5:7b

**Initial implementation (Jan 2026):** The first LLM integration (PR #26) was a minimal `local_llm.py` that called whatever Ollama model was available. No structured output, no field descriptions; just a prompt asking for JSON.

**BioMistral experiment (Feb 2026):** We added BioMistral (a biomedical fine-tune of Mistral 7B) in a separate `biomistral_llm.py` with the expectation that domain-specific pretraining would help with biological terminology and study design language. In practice, BioMistral was more likely to hallucinate plausible-sounding stomach count numbers than a general-purpose model. Its training distribution skewed toward clinical/genomics literature, not field ecology surveys, so the domain-specific advantage never materialized.

**Switch to qwen2.5:7b (Mar 2026):** qwen2.5:7b consistently followed the structured output format more reliably and returned fewer null fields. We decided to switch to qwen2.5:7b after a head-to-head comparison on the same set of test papers. This model was then standardized across all entry points.

**What to watch if you try a different model:** The pipeline uses Ollama's structured output feature (JSON schema enforcement), which not all models handle equally well. The retry logic in `llm_client.py` helps absorb transient failures but won't fix a model that systematically ignores the schema. Test any new model against the papers in the `data/` test set before committing to it.

---

## 3. Text Preprocessing: Getting the Right Text to the LLM

This was the area that evolved the most throughout the project. The core problem: a typical ecology paper is 8,000–25,000 tokens, but a 7B model's context window is 4,096 tokens by default. We can't just truncate at the limit; we need to send the *right* content.

### 3a. Text Cleaning

The raw PyMuPDF output for many papers contains a lot of noise that confuses the LLM: DOI strings scattered through paragraphs, numbered references mixed into body text, figure captions, table footnotes. We added a text cleaner (`src/io/text_cleaner.py`) that strips these before the text reaches either the classifier or the LLM.

### 3b. OCR for Scanned Papers

Many ecology papers from the 1970s–1990s are only available as scans. PyMuPDF fails silently on these, returning an empty or near-empty string. We added Tesseract as a fallback triggered when the PyMuPDF extraction yields fewer than a threshold number of characters. Later (PR #37, Feb 2026) we added image denoising before the OCR pass to improve quality on low-contrast scans.

**Note for future teams:** OCR caused workers to hang when running parallel PDF processing. A `--no-ocr` flag was planned as an escape hatch but was not implemented in the final pipeline. If workers hang on a batch that includes scanned PDFs, reduce `--workers` to `1` as a workaround.

### 3c. Section Priority Ranking

We tried several approaches before settling on the current one:

- **First attempt:** Simple character-limit truncation from the top of the document. This sent the abstract + introduction to the LLM and missed the Results section where the actual stomach counts are reported.
- **Second attempt:** Page-priority selection. We tried to predict which pages were most informative. This was removed as dead code (PR #63) because it never worked reliably across paper formats.
- **Final approach:** Section-boundary detection via regex, with an explicit priority ranking: Abstract (0) → Results (1) → Methods/Study Area (2) → Tables (3) → Introduction (4) → Discussion (5). Sections are included in priority order until the character budget is exhausted. The entire section drop list (references, acknowledgments, appendices, supplementary materials) is stripped before this ranking runs. This is implemented in `src/extraction/llm_text.py:extract_key_sections()`.

### 3d. Paragraph Scoring

On top of section priority, we added paragraph-level filtering: paragraphs that contain no positive signal words (stomach, predator, prey, diet, feeding, etc.) and no negative signal words are dropped. This was particularly important for Methods sections, which often contain several paragraphs about study site geography or permit numbers that are irrelevant to extraction.

---

## 4. Table Extraction

Ecology papers frequently report stomach counts in tables rather than prose. We added `camelot-py` for structured table extraction (PR #30, Jan 2026). The first merge was reverted the same day because camelot's Ghostscript dependency wasn't installed in the CI environment and the tests failed. It was re-merged (PR #32) after fixing the CI environment and the Ghostscript dependency.

**The current behavior:** Camelot extracts tables to plain text, which is then injected into the text stream at priority level 3 (after abstract, results, and methods sections). This matters because stomach count tables are often the single most information-dense part of a paper.

**Known limitation:** Camelot works well on native PDFs. Scanned papers have no embedded table structure, so camelot gets nothing useful from them. Those papers fall back to whatever OCR extracted.

---

## 5. LLM Prompt Evolution

The system prompt went through several rewrites:

**Initial prompt:** A short description of the task with a list of field names. Return rates were poor; the LLM frequently returned `null` for fields that were present in the text.

**Added field descriptions (PR #36, Feb 2026):** Each field in the Pydantic schema got an explicit description including common phrasings found in the literature (e.g., "stomachs with no contents", "N empty", "N with no prey" for `num_empty_stomachs`). This alone improved the extraction rate substantially.

**Added few-shot examples:** The prompt includes concrete worked examples of input text and expected JSON output. This helped the most with `study_date`, where models were frequently returning publication years instead of collection years until we added an example that made the distinction explicit.

**Rewrote for diverse sampling methods (Mar 2026):** The original prompt assumed stomach dissection as the sampling method. Many papers use stomach pumping, scat analysis, or regurgitation. The prompt was rewritten to cover these cases after we noticed systematic null returns on a set of seabird diet papers that used regurgitation sampling.

---

## 6. Pydantic Schema Design

The `PredatorDietMetrics` model in `src/extraction/models.py` evolved significantly.

**Original:** A plain Python dataclass with no validation.

**Added Pydantic (PR #26 era):** Basic type annotations. No constraints.

**Tightened constraints (Feb–Apr 2026):**
- `species_name`: regex enforcing binomial format (`^[A-Z][a-z]+(\s[a-z]+)*$`). This prevents the LLM from returning common names like "Northern pike" instead of *Esox lucius*.
- `study_date`: regex enforcing `YYYY` or `YYYY-YYYY` format only.
- `sample_size`: must be positive (> 0), not just non-negative.
- `num_empty_stomachs` and `num_nonempty_stomachs`: non-negative integers.

**Auto-reconciliation:** If `num_empty_stomachs` and `num_nonempty_stomachs` are both present, `sample_size` is automatically set to their sum via a `@model_validator`. This prevents subtle inconsistencies when the LLM extracts counts from a table but reads the stated sample size from a different paragraph.

---

## 7. Pipeline Architecture Decisions

### 7a. Two-Stage Gating

The classify-then-extract structure was intentional from the start. Running every PDF through the LLM would be prohibitively slow at scale. LLM extraction takes 30–90 seconds per paper depending on length and hardware. The XGBoost classifier reduces that to under a second per paper. The confidence threshold (`--confidence-threshold`, default 0.70) lets users trade recall for throughput.

### 7b. Parallel Workers

We added `--workers` to enable parallel PDF processing across CPU cores. The implementation uses Python `multiprocessing`. Two issues we found during testing:
1. Tesseract's subprocess model doesn't play well with Python's multiprocessor on Windows. If workers hang, fall back to `--workers 1`.
2. SpellChecker (used during text cleaning) was being instantiated per-PDF. Moved to a module-level singleton.

### 7c. LLM Retry Logic

Ollama occasionally returns an empty response or times out, especially under memory pressure when the model is being evicted and reloaded. We added exponential backoff retry logic (3 attempts, 2-second base delay) in `llm_client.py`. The timeout per call is 120 seconds. These values were chosen empirically. On a machine with 16 GB RAM, 120 seconds is enough for qwen2.5:7b on papers up to ~15,000 characters.

---

## 8. Dataset & Training Pipeline

**Dataset source:** The [FracFeed global database](https://github.com/marknovak/FracFeed_DB) maintained by Mark Novak's lab. Contact Mark before retraining, as the labeled set has grown over time and the version used for the current model is not pinned in the repo.

**Two training modes:**
- `--local`: Fastest. Reads PDFs from a local folder with `useful/` and `not-useful/` subdirectories.
- `--api`: Fetches the dataset from Google Drive via the service account in `.env`. Added (PR #19) because not all team members had the full dataset locally. **Important:** local mode is significantly faster for iteration because the API mode pulls several GB over the network each time.

**Retraining:** See the Retraining section of `CONTRIBUTING.md` for the current steps. The short version: add labeled `.txt` files to `data/processed-text/` and update `data/labels.json`, then run `python -m src.classifier.train_model`.

---

## 9. Repository Structure Evolution

The `src/` directory was reorganized once near the end of the project (PR #64, May 2026), as the original flat structure became hard to navigate:

| Old path | New path |
|---|---|
| `src/model/` | `src/classifier/` |
| `src/llm/` | `src/extraction/` |
| `src/preprocessing/` | `src/io/` |
| `classify_extract.py` (root) | `src/pipeline/classify_extract.py` |
| `extract-from-txt.py` (root) | `src/pipeline/extract_from_txt.py` |

`requirements.txt` was also removed at this point in favor of `pyproject.toml` as the single source of dependency truth.

---

## 10. Things That Didn't Work Out

**Page-number tracking in extraction output:** We briefly added a `page_number` field to `PredatorDietMetrics` to record which page the stomach counts came from. This was reverted because the section-priority extraction splices content from multiple pages into a single string, making per-field page attribution unreliable. The `provenance` concept was replaced by the section-source labeling in the final output.

**Structured output via OpenAI-compatible Ollama endpoint:** Early experiments called Ollama via its OpenAI-compatible REST API with response_format enforcement. We moved away from this to the native `ollama` Python library because the REST path had worse JSON schema compliance in the versions we tested, and the Python client gave us easier access to timeout and retry hooks.

**BioMistral:** Covered in section 2 above.


---

## 11. Known Limitations and Suggested Future Work

- **Single-paper extraction only:** The pipeline extracts one `PredatorDietMetrics` record per paper. Many diet survey papers report results for multiple species or across multiple years. A chunked multi-record extraction mode is partially stubbed in `src/extraction/chunked_extraction.py` but was not completed.
- **Provenance is coarse:** The current output records classifier confidence and extraction timestamps but doesn't link each extracted field back to a specific sentence. Adding sentence-level provenance would make downstream QA much faster.
- **Classifier retraining is manual:** There's no automated pipeline to re-train the classifier when new labeled data is added. A CI trigger on changes to `data/labels.json` would help.
- **OCR quality:** For very low-quality scans (pre-1970s papers), the current pipeline often produces unusable text even with denoising. A dedicated OCR quality check before extraction would reduce false negatives.
