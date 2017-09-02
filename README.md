## Code and Data for the NMF for Overlapping Clustering of Customer Datasets Project

### Intro to each folder
- `experiment` stores the code used for generating the results in Chapter 3 and Chapter 6 of the dissertation. `evaluation_metric_scores.py` returns the metric scores of Semi-NMF, Convex-NMF, Fuzzy C-means with m=1.1 and m=2. `evaluation_residual.py` returns the residual and absolute second order derivative of the residual of the four algorithms. `print_nearest_5_words.py` returns the nearest 5 words of each basis vector of the four algorithms. The dataset and vector representation for each code are entered by user. `test_Soft_BCubed_metrics.py` shows the two examples of Soft BCubed Metrics in Chapter 3.

- In `experiment`, the subfolder `pre-processed_dataset` stores the vector representations of the datasets. The subfolder `vocabulary` stores the vector representations of each word in the vocabulary. Note that due to the upload policy of GitHub, some large files are zipped, so please make sure to unzip all files in the two subfolders before running the code in `experiment`.

- `metrics` stores the code of Extended Silhouette index, Soft BCubed metrics, Extended BCubed metrics and Purity metrics.

- `NMF_algorithm` stores the code of Semi-NMF and Convex-NMF.

- `code_for_pre-processing_dataset` stores the pre-processed datasets of Amazon Review, Yelp Review and Maluuba Frames, as well as codes for encoding them to vector representations.

### How to run

Simply run each file directly using the command-line interface or an IDE. Then enter corresponding dataset name, vector representation name or number of clusters as instructed by the printed message.
