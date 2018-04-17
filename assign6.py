#############################
####   Anurag Banerjee   ####
####       CS7030        ####
####    Assignment 6     ####
#############################
from model import *
from feature import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.datasets.samples_generator import make_blobs
# from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


'''
Step 1: create a data set of mixed documents from the newsgroup dataset, say take 30 from each
Step 2: apply K-means with K = 20
Step 3: Use pca/lda and plot scatter plot of results to visualize the result
'''


def drawClustersPCA(ndim_x, y, k):
    """
    In this function we will take the array of points, their y axis labels and scatter plot them on two dimension
    using Principal Component Analysis to perform dimensionality reduction
    :param ndim_x: The document vectors
    :param y: the labels for the documents
    :return: None
    """
    pca = sklearnPCA(n_components=2)    # 2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(ndim_x))
    colors = cm.rainbow(np.linspace(0, 1, k))
    for i, c in enumerate(colors):
        plt.scatter(transformed[y == i][0], transformed[y == i][1], c=c) # label='Class '+str(i+1), c=c)

    # plt.legend()
    plt.show()


def drawClustersLDA(ndim_x, y, k):
    """
    In this function we will take the array of points, their y axis labels and scatter plot them on two dimension
    using Linear Discriminant Analysis to perform dimensionality reduction
    :param ndim_x: The document vectors
    :param y: the labels for the documents
    :return: None
    """
    lda = LDA(n_components=2)   # 2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(ndim_x, np.ravel(y)))

    colors = cm.rainbow(np.linspace(0, 1, k))
    for i, c in enumerate(colors):
        plt.scatter(lda_transformed[y == i][0], lda_transformed[y == i][1], c=c) # label='Class '+str(i+1), c=c)

    # Display legend and show plot
    # plt.legend(loc=3)
    plt.show()


def showDocClusters(df, k):
    '''
    for i in range(k):
        print('cluster '+str(i+1)+' has following docs: ')
        df2 = df['DocList'].loc[df['ClusterLabel'] == i]
        print(df2)
    '''
    for cluster_no, docDF in df.groupby('ClusterLabel'):
        print('cluster ' + str(cluster_no) + ' has following docs: ')
        print(docDF['DocList'])


def main():
    # Change the path to the training data directory
    data = readfiles1('workdata')
    '''
    # Initialize the model and preprocess
    bow = BagOfWordsFeatureExtractor()
    bow.preprocess(data)

    # Extract fetures and create a numpy array of features
    X_data_bow = bow.extract(data)

    model1 = Kmeans()
    #model1.train(X_train_bow, Y_train_bow, lr, reg_const)
    labels = model1.cluster(X_data_bow, k=5, n_iter=25)
    '''

    # Initialize the model and pre-process
    tfidf = TfIdfFeatureExtractor()
    print("Pre-processing...")
    tfidf.preprocess(data)
    print('...done')

    # Extract fetures and create a numpy array of features
    print('Extracting features...')
    (X_data_tfidf, doclist) = tfidf.extract(data)
    df = pd.DataFrame(doclist, columns=['DocList'])
    print('...done')

    model2 = Kmeans()
    k = 20
    n_iter = 40
    print('forming '+str(k)+' clusters, in '+str(n_iter)+' iterations...')
    labels = model2.cluster(X_data_tfidf, k, n_iter)
    df['ClusterLabel'] = labels.flatten().tolist()
    print('...done')
    # print(len(labels))
    # print(labels)
    drawClustersLDA(X_data_tfidf, labels, k)
    # drawClustersPCA(X_data_tfidf, labels, k)
    showDocClusters(df, k)


if __name__ == '__main__':
    main()
# EOF
