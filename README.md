## Code for the NMF for Overlapping Clustering of Customer Datasets Project

### Intro to each folder
- `code_for_pre-processing_dataset` stores the pre-processed datasets of Amazon Review, Yelp Review and Maluuba Frames, as well as codes for encoding them as vector representations

- `experiment` stores the code used for generating the results in Chapter 3 and Chapter 6 of the dissertation. The two subforders `pre-processed_dataset` and `vocabulary` stores the vector representations of the datasets and the vector representation of each word in the vocabulary respectively.

- `metrics` stores the code of Extended Silhouette index, Soft BCubed metrics, Extended BCubed metrics and Purity metrics.

- `NMF_algorithm` stores the code of Semi-NMF and Convex-NMF.

### How to run

Simply run each file directly using command-line interface or an IDE. Then enter corresponding dataset name, vector representation name or number of clusters as instructed by the printed message.