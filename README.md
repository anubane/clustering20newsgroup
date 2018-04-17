## Clustering 20 newsgroup data
Applying K-Means algorithm to the 20 newsgroup data (link: )
* **Step 1**: From the already classified data, I have selected 30 from each group and mixed them under directory `workdata`
* **Step 2**: Using tf-idf model, extract features for the documents and form the document vectors
* **Step 3**: Apply K-Means algo to these n-dim doc vectors
* **Step 4**: The labels generated for the cluster membership and the doc vectors are passed to the LDA function
* **Step 5** The linear discriminant analysis process reduces the n-dim to 2-dim and plots the clusters via a scatter plot (`Cluster_Visualization.png`)
* **Step 6** The doc list of each cluster is also output (sample available in `Clusters_of_Docs.txt`)
