# FracFeedExtractor - LLMs for the fraction of feeding predators

## Project Description
This project contributes to validating a novel metric of predator-prey interactions to inform ecosystem-based resource management and ecological theory.  It does so by using a global database of predator diet surveys to train large language models for the purpose of identifying additional publications and extracting key data to overcome the limitations that have hindered the empirical validation of the new metric thus far.


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

## Team Members
- Mark Novak – Project Owner/Lead  
- Sean Clayton – Contributor  
- Zahra Zahir Ahmed Alsulaimawi – Contributor  
- Raymond Cen – Contributor  
- Bradley Rule – Contributor

License: Pending partner confirmation

## Documentation
- [Contributing Guide](documentation/CONTRIBUTING.md)
- [Pipeline Architecture Diagram](documentation/architecture.png)
