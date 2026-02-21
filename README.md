# TabulaML

TabulaML is a modular, extensible AutoML-inspired platform for structured (tabular) data, designed to demonstrate production-oriented machine learning engineering practices.

The project focuses on building reproducible, maintainable, and well-architected ML pipelines rather than providing a black-box solution. It reflects real-world engineering principles such as modular design, separation of concerns, experiment orchestration, and backend-driven model training.

This repository is part of my professional portfolio and showcases my approach to building scalable ML systems with clean architecture.

---

## Executive Summary

TabulaML enables users to:

- Upload a tabular dataset.
- Automatically inspect and analyze feature characteristics.
- Generate structured preprocessing pipelines.
- Execute multiple modeling experiments by stage.
- Compare evaluation metrics in a systematic way.
- Select and serialize the best-performing model.

The system is backend-driven (Python-based), with a clear separation between orchestration logic, experimentation modules, preprocessing, and evaluation.

---

## Key Engineering Highlights

- Modular architecture with clear stage separation.
- Builder pattern for pipeline construction.
- Experiment definition abstraction.
- Automatic feature inspection and noise detection.
- Extensible experiment registry.
- Reproducible ML workflows.
- Backend-first design ready for API exposure.
- Designed with microservice evolution in mind.

---

## System Architecture

### High-Level Design

The system is divided into two logical layers:

### 1. Frontend (Conceptual Layer)

Responsible for:

- Collecting user configuration.
- Guiding the ML workflow stages.
- Displaying experiment results.
- Sending structured configuration payloads to the backend.

No training or preprocessing is performed in the frontend.

---

### 2. Backend (Python Core Engine)

Responsible for:

- Data inspection.
- Feature validation and noise detection.
- Preprocessing pipeline construction.
- Experiment orchestration.
- Model training and evaluation.
- Metrics comparison.
- Model serialization.

The backend is structured around independent modules that interact through clearly defined interfaces.

---

## Core Workflow

1. Dataset ingestion.
2. Automatic data inspection:
   - Data types
   - Cardinality analysis
   - Constant columns
   - High-cardinality identifiers
   - Potential noisy features
3. Preprocessing pipeline construction.
4. Execution of experiments by stage:
   - No feature selection
   - ElasticNet-based selection
   - Random Forest
   - Other extensible models
5. Metrics evaluation and comparison.
6. Best model selection.
7. Model export.