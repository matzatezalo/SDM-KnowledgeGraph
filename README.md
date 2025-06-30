# SDM Project 2: Knowledge Graph & Embeddings Project

This repository contains all code, data transforms, SPARQL queries, and embedding-training scripts for building and exploiting a **Knowledge Graph** of academic publications.  

Project report **SDM_Project_Report.pdf** contains all information about our work.

1. **Ontology (TBox)**: schema for papers, authors, venues, topics, etc.  
2. **Data Lifting (ABox)**: Python scripts to convert JSON into RDF triples.  
3. **Querying & Analytics**: SPARQL queries for RDFS reasoning and graph-analytic metrics.  
4. **Knowledge Graph Embeddings (KGEs)**:  
   - Import/export to TSV for PyKEEN, generate triples  
   - Train and evaluate TransE, DistMult, ComplEx  
   - Link-prediction experiments  
   - Clustering & downstream tasks
  
---
