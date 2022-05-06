
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib.pyplot as plt 
from numpy.linalg import norm
from sklearn.cluster import KMeans, SpectralClustering
import seaborn as sns
import sklearn
sns.set()
from scipy.stats import norm

def readFile(fileName):
    """
    This function will read the text files passed & return the list
    """
    fileObj = open(fileName, "r") #opens the file in read mode
    words = fileObj.read().splitlines() #puts the file into a list
    fileObj.close()
    return words

"""# Reading N-Grams
- After looking at the top 100 results produced in Collocations step, I concluded that frequency, t-test & likelihood test performs good after filtering & give mostly similar results.
- Whereas, PMI & Chi-sq test gave similar & good result without even applying filter, but I still applied filter on both of the methods, since although these methods are giving good results, but still considering some "let us" "last updated" like occurences as meaningful.
- Applying filter might haven't deleted such sense less occurrences, but yeah it has reduced their preference in the list.
- I have  visualized each of the filtered method, and I get to see other than frequency, all the methods are giving meaningful & similar clustering results.
- Because the order of preferences is changing in each method otherwise generated n-grams are mostly similar.
- So, Here I have used filtered likelihood bigrams & trigrams list
"""

def read_nGrams():
    """
    This function will read bigrams & trigrams and 
    return combined list of bigrams & trigrams.
    """
    # read  bigrams 
    original_bigram = readFile("bigram.txt")
    # read trigrams
    original_trigram = readFile("trigram.txt")

    # Combined list of bigrams & trigrams
    n_grams_to_use = []
    n_grams_to_use.extend(original_bigram)
    n_grams_to_use.extend(original_trigram)
    return n_grams_to_use
n_grams_to_use = read_nGrams()

# split each n-gram into separate words
def split_nGrams(n_grams_to_use):
    ngrams_splited = [each.split() for each in n_grams_to_use]
    return ngrams_splited
ngrams_splited = split_nGrams(n_grams_to_use)
len(ngrams_splited)

"""# Step1: functions to get n-grams vectors"""

def average_word_vectors(list_words, model, vocabulary, num_features):
    """
    This function will take each tokenized sentence having bigrams or trigrams, 
    model = the mapping_of_word_to_vector dictionary, vocabulary = unique set of keys(words) present in model,
    num_features = 50
    
    This function will return the average of feature vector for each word present in list_words.
    """
    # Created array of zeros (type float) of size num_features, i.e., 50.
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    # Put it in try block so that if any exception occur, it will be dealt by below exception block.
    try:
        # Check if word is in passed list_of_words or not.
        for word in list_words:
            # Check if word is in general vocabulary or not (the unique set of words in word embedding).
            if word in vocabulary: 
                # Increment number_of_words
                nwords = nwords + 1
                # add vector array of corresponding key in model which matches the passed word.
                feature_vector = np.add(feature_vector, model[word])

        if nwords:
            # Take average of feature_vector by dividing with total number of words
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector
    
    except:
        # If the exception occurs, while the word isn't found in vocabulary, it will return the array of zeros
        return np.zeros((num_features,),dtype="float64")
    

    
def averaged_word_vectorizer(corpus, model, num_features):
    """
    This function is taking corpus of bigrams & trigrams, w2v mappings, num of features as a input arguments.

    and returning array of features after taking average using average_word_vectors() function.
    """
    # Get the unique keys out of word_to_vector_map dictionary.
    vocabulary = set(model.keys())
    # Call function average_word_vectors which is returning with averaged vectors for each word in tokenized sentence.
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in ngrams_splited]
    return np.array(features)

#  function to read glove vectors from text file
def read_glove(glove_path):
    """
    This function will read glove data from text file and do the following:
    1. prepare dictionary of words and vectors
    2. prepare dictionary of words and index
    3. prepare dictionary of index and words
    """
    # Read word_embedding file stored on glove_path specified.
    with open(glove_path, 'r', encoding='utf-8')as inp_file:
        
        words = set()
        word_to_vec_map = {}
        
        # For every line in embedding file which contains the word & the corresponding vector.
        for line in inp_file:
            # convert each line in embedding file to a list of elements.
            line = line.strip().split()
            # Get first element of the list, i.e., word of each list.
            curr_word = line[0]
            # Add the distinct set of words.
            words.add(curr_word)
            # Create dictionary that will map current word to that of it's vector representation.
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
        i=1
        words_to_index = {}
        index_to_words = {}
        # For every word in sorted dictionary of words
        for w in sorted(words):
            # map index to each words
            words_to_index[w]=i
            # map words to each index
            index_to_words[i]=w
            i += 1
        
        return words_to_index, index_to_words, word_to_vec_map

# load glove vectors from pre-trained model domain dataset
glove_path = r"Generating_nGrams\Text Clustering\domain_embeddings.txt"
new_words_to_index, new_index_to_words, new_word_to_vec_map  = read_glove(glove_path)

# Create word to vector averaged feature array
w2v_feature_array = averaged_word_vectorizer(corpus=ngrams_splited, model=new_word_to_vec_map,
                                             num_features=50)
w2v_feature_array.shape

# Create Dataframe 
df = pd.DataFrame(w2v_feature_array)
df.index = n_grams_to_use
df.head()

"""# Transpose
- I have applied further steps on the transpose, bcz It is giving somewhat good separable clusters.
- One thing I want to know from you Karan, as I know it's not feasible or practical to feed such dataset to any ML algo where the no. of columns are greater than no. of rows.
- But in my case, I am getting good result. Is this the correct way??
- Although, I am applying PCA in further steps to reduce dimensionality reduction
"""

Transposed_Dataset = df.T
Transposed_Dataset.head()

"""# Scale the Dataset
- Since, We are working on un-supervised learning and it works bad on low data. 
- so there is a need to scale data before feeding it to k-means algo.
"""

# Used preprocessing module of sklearn library to scale data.
X_scaled = preprocessing.scale(Transposed_Dataset)
len(X_scaled)

"""# Standardize the Data
- To ensure internal consistency of the data means  each data type will have the same content and format. 
- Standardized values are useful for tracking data that isn't easy to compare otherwise.
"""

# I have used StandarScaler() & fit_transform() function of sklearn library to standardize features.
X_std = StandardScaler().fit_transform(X_scaled)

"""# Check uniformity of Dataset
- I am checking uniformity of both the datsets using KL divergence test. 
- I am using KL test just for having an idea of distributions, we are not comparing distributions here.
- Extras
- The KL divergence is zero, if two distributions are equal.
- The KL divergence is positive, if two distributions are different.
"""

df_variance = df.var() # Calculating variance for each feature of dataframe
df_mean = df.stack().mean() # Calculating mean for each feature of dataframe
trans_mean = Transposed_Dataset.stack().mean() # Calculating mean for each feature of transpose
trans_variance = Transposed_Dataset.var() # Calculating variance for each feature of transpose
df_std = df.stack().std()   # Calculating standard deviation for each feature of dataframe
trans_std = Transposed_Dataset.stack().std() # Calculating standard deviation for each feature of transpose
        
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def df_normal_dist():
    x = np.arange(-10, 10, 0.001)
    dp = norm.pdf(x, df_mean, df_std) 
    dq = norm.pdf(x, 0, 1) 
    # Taking KL divergence b/w Dataframe distribution & normal distribution.
    plt.title('KL(P||Q) = %1.3f' % kl_divergence(dp, dq))
    plt.plot(x, dp)
    plt.plot(x, dq, c='red')
df_normal_dist()

def trans_normal_dist():
    x = np.arange(-10, 10, 0.001)
    tp = norm.pdf(x, trans_mean, trans_std) 
    tq = norm.pdf(x, 0, 1)
    # Taking KL divergence b/w Dataframe's transpose distribution & normal distribution.
    plt.title('KL(P||Q) = %1.3f' % kl_divergence(tp, tq))
    plt.plot(x, tp)
    plt.plot(x, tq, c='red')
trans_normal_dist()

def df_trans_dist():
    x = np.arange(-10, 10, 0.001)
    p = norm.pdf(x, df_mean, df_mean )
    q = norm.pdf(x, trans_mean, trans_std)
    # Taking KL divergence b/w Dataframe's  distribution & Dataframe's transpose distribution.
    plt.title('KL(P||Q) = %1.3f' % kl_divergence(p, q))
    plt.plot(x, p)
    plt.plot(x, q, c='red')
df_trans_dist()

"""# Conclusion of applying Uniformity Distribution:
- I have first compare plots of each dataset with that of normal distribution, bcz I was getting bell shaped distributions for each Dataset.
- From above one can easily conclude that the Dataset is not distributed uniformly since we are getting some bell shaped curves.
- So, We can apply k-means algorithm easily.

# Step2: Peform clustering
## Write your function to generate clusters from the feature array of input data.
"""

sklearn_pca = PCA(n_components = 2) # Using PCA to remove cols which has less co-relation
Y_sklearn = sklearn_pca.fit_transform(X_std) #fit_transform() is used to scale training data to learn parameters such as 
# mean & variance of the features of training set and then these parameters are used to scale our testing data.
# As concluded using Elbow Method.
n_clusters = 2
kmeans = KMeans(n_clusters= n_clusters, max_iter=400, algorithm = 'auto')# Partition 'n' no. of observations into 'k' no. of clusters. 
fitted = kmeans.fit(Y_sklearn) # Fitting k-means model  to feature array
prediction = kmeans.predict(Y_sklearn) # predicting clusters class '0' or '1' corresponding to 'n' no. of observations

"""# Elbow Method to select optimal number of clusters"""

def elbow_method(Y_sklearn):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(1, 7)  # Range of possible clusters that can be generated
    kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters] # Getting no. of clusters 

    score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))] # Getting score corresponding to each cluster.
    score = [i*-1 for i in score] # Getting list of positive scores.
    
    plt.plot(number_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Method')
    plt.show()
elbow_method(Y_sklearn)
# Optimal Clusters = 2

"""# Conclusion after applying Elbow Method:
- As shown in the above figure, the knee of a elbow curve signifies the cut-off point to be considered as optimal no. of clusters to be chosen.
- So, the optimal no. of clusters is 2
"""

def kmeans_clustering(Y_sklearn, fitted):
    """
    This function will predict clusters on training set and plot the visuals of clusters as well.
    """

    plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction ,s=50, cmap='viridis') # Plotting scatter plot 
    centers2 = fitted.cluster_centers_ # It will give best possible coordinates of cluster center after fitting k-means
    plt.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6);
    # As this can be seen from the figure, there is an outlier as well.
kmeans_clustering(Y_sklearn, fitted)

"""# Treating Outliers

### K-means Clustering is sensitive to outliers, bcz mean can easily be influenced by outliers. When I first applied k-means clustering without removing outliers, I am getting this kind of figure.
![image.png](attachment:image.png)
"""

def get_top_features_cluster(X_std, prediction, n_feats):
    # Get unique labels, in this case {0,1}
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # Get indices for each feature corresponding to each cluster.        
        x_means = np.mean(X_std[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = n_grams_to_use
        best_features = [(features[i], x_means[i]) for i in sorted_means] # Retrieve corresponding best features to that of best scores.
        Df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(Df) # append both the Dataframes to a list
    return dfs
dfs = get_top_features_cluster(X_std, prediction, 20)

plt.figure(figsize=(8,6))
sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[:25][0]) # Get top 25 rows of 1st Dataframe

plt.figure(figsize=(8,6))
sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[:25][1]) # Get top 25 rows of 2nd Dataframe

# Creating csv files of top 20 features extracted
for i, df in enumerate(dfs):
    # Using enumerate() to add counter to an iterable list dfs.
    df.to_csv('DF_'+str(i)+'.csv')

def plot_features(dfs):
    """
    This function will print combined bar graphs for all the possible clusters.
    """
    fig = plt.figure(figsize=(14,12))
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.set_title("Cluster: "+ str(i), fontsize = 14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#40826d')
        yticks = ax.set_yticklabels(df.features)
    plt.show();
plot_features(dfs)