# PatternMind 321071 
Group members: 

Student 1: Galentino Mattia, 324331

Student 2: Vertunni Alessandro, 310231

Student 3: Viallard Alexandre, 321071

## Introduction

Our group worked on the PatterMind task. The goal of this task was to work with a database consisting of subfolders each with a varying degree of number of images. Firstly, in this project we classified the different images based on their labels (subfolders). Secondly, we analysed each category to determine similarities and connections among categories and subcategories. Here, our secondary goal was to unconver the semantic relationships through unsupervised learning methods. In the next section we will go over the methods of our analysis.

## Methods
To accomplish the PatterMind task, we decided to work on Google Colab. That way, we could share the files online and each member could contribute without having to manually send the file to each other each time. Furthermore, Colab is an environment we were already familiar with from our previous computer science related courses.

Before running the code, we uploaded the dataset on Google Drive and linked it to Colab's folders. In this way, it was easily accesible.

We used a GPU called T4 GPU, selected from Colab's runtimes for a faster performance.

The first thing we did in the code was importing the libraries we used in our task. The libraries imported are the following:
* numpy
* matplotlib
* seaborn
* pandas
* os
* sklearn (for machine learning purposes) from which we import:
    * train_test_split
    * confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    * KMeans
    * PCA
* tensorflow (for neural networks design) from which we import:
    * Sequential, Model
    * Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    * ImageDataGenerator
    * to_categorical

**Exploration of dataset**

First of all, we had to explore the dataset performing exploratory data analysis (EDA). The reason for this is to gather as much information as possible on the dataset that we are dealing with. The first thing we did was to count the number of subfolders we had in the main dataset. The name of these subfolders would become the labels used in the models we would build for prediction and also the labels we would use for semantic analysis. The next step in our exploration was to count the number of images per subcategory (or label). 

Below, as shown in figure 1, we can see a barplot displaying the number of images per subcategories for a sample of 50 observations.

![figure_1](./figure1.png)

In figure 1, the sample of 50 observations is made up of the 10 subcategories with the highest number of images within their folder and the 10 lowest. The remaining 30 observations were randomly selected from the sample without replacement.

**Data Loading**

At this point we had an idea of how the dataset was subdivided, so it was time to import the images into an array. To do so, we utilized Keras's built-in function "image_dataset_from_directory". Note that we did not split the dataset in training, validation and test yet. We will do so later, taking into account class imbalance. We then converted the imported images into a NumPy array to facilitate data manipulation.

Images were also resized to 28x28 pixels and normalized to fit in a range of [0,1].

**Balancing**

We then proceeded to generate more images for the unbalanced classes. We chose to separate the dataset like this: 37 test images for all classes and 147 train images for all classes. Note that the two numbers sum up to 184, which is double the lenght of the median of number of images per class. We chose this number because it sits right in the middle of all observations (the arithmetical mean is sensible to outliers) and we doubled the number so that the models could have more images to train without adding too many leading to skewed results due to an overly large image augmentation of the training set.

Test set contains only original images. Image augmentation is exclusively performed on the training dataset, so that the final model evaluation is performed solely on original images. Image augmentation followed this criteria: after having picked the 37 unique original test images, if the class of the training set had 147 original images, they would be used to train the model. Instead, if the class had less than 147 original images to perform training, then we would generate the necessary number of images to reach the specified number through augmentation. 

**Model outlining**

We then proceeded to define our models. Since the first part of the task consists in image classification, we decided that neural networks would be the most appropriate choice. Our first choice was CNN since it is known to perform effectively image classification, as it can recognize spatial patterns through kernels. Our second choice was ANN, even though it does not capture spatial patterns as effectively as CNN. Nonetheless, we chose this model to compare a simple neural network to a convolutional neural network. To make this comparison, we will deploy different metrics such as accuracy. As our third model, we chose logistic regression. Logistic regression was our baseline, relatively simple model. Comparing Logistic Regression, ANN, and CNN reveals how linear models, non-linear learners, and spatial feature extractors differ in their ability to capture complexity and structure in the data.

**Semantic analysis**

For the last part of the project, we wanted to investigate whether there were semantic relationships between classes. We hoped to find semantic connections between image features. To achieve this goal, we got the feature vectors from the CNN model and reduced them employing to PCA. We then plotted the clustering using KMeans. Next, we manually examined and uncovered the main theme of each cluster as well as micro-themes within the cluster itself considering semantic relations. 

## Experimental Design

In this section we will go over all experiments we conducted to demonstrate the target contributions of our code. We will begin with how we chose the best model design.

**Model design**

To find the best settings of hyperparameters, we performed KFolds cross-validation. We specified different configurations for CNN, ANN and logistic regression.

The hyperparameters we decided to tweak are the following: 
- CNN:
    - Convolution Filters
    - Dropout Rate
    - Dense units
    - Learning rate
    - Batch size

- ANN:
    - Number of hidden layers
    - Dropout
    - Learning rate

For both ANN and CNN we chose those parameters because we found them to be among the most essential in architecture design and also the ones we were most familiar with from previous studies.

- Logistic Regression:
    - C regularization  (to penalize overfitting)
    - Maximum number of iterations

We evaluated hyperparameters based on how they performed on the validation set.

Afterwards, we evaluated the best models on the test set. We selected a set of metrics that would allow us to compare performance. We considered the following:

- Accuracy
- Precision
- Recall
- F1 score

Accuracy measures the proportion of correct predictions out of all predictions, whereas precision and recall give a more detailed evaluation of false positives and false negatives. The F1 score balances precision and recall.

Our baseline experiments were designed to compare different model families and determine whether model complexity improves performance in a meaningful way. Logistic regression was considered as a simple linear model. ANN introduced nonlinearity. CNN introduced spatial filters to extract feature patterns from images.

The logic of these comparisons was straightforward: if the CNN outperforms ANN and logistic regression, it confirms that spatial feature extraction is useful for our images.

We trained and evaluated all three models described above, and below we report the most relevant metrics.

| Model                 | Accuracy  | Precision | Recall | F1 score |
|-----------------------|-----------|-----------|--------|----------|
| CNN                   | 20,72%    | 20,06%    | 20,72% | 19,32%   |
| ANN                   | 2,45%     | 0,75%     | 2,45%  | 0,77%    |
| Logistic Regression   | 9,74%     | 7,99%     | 9,74%  | 8,48%    |

We observe: CNN achieved the best accuracy and F1 score, logistic regression was the second best performer after CNN, while ANN was the model that performed the worst. We attribute the lower accuracy of ANN due to model complexity. Our ANN has 658,665 trainable parameters and the dataset contains only 150 samples, and so this means that the model is more complex than the data. The logistic regression model is simpler and has less parameters, so it is learns better.

![figure 2-3](.\figure2-3.png)
**figure 2-3**

Further analysis based on figure 2 and 3:
* The CNN reaches about 19.6% accuracy and shows steady improvements in its learning curves, but the validation accuracy consistently is above the training accuracy, which is expected when augmentation is applied only to the training data or when the training images are simply harder than those in the validation split.
* The ANN remains extremely low at around 1.2% accuracy, and its validation accuracy also exceeds its training accuracy, reinforcing that the model is not learning effectively and that the training data is likely more challenging due to augmentation or more likely the ANN i.
* Logistic Regression reaches about 9.7 percent accuracy and still performs better than the ANN even though both use raw pixel inputs. This happens because Logistic Regression applies a single linear decision boundary across all pixel values at once, which can sometimes pick up simple global patterns like overall brightness or silhouette shape. The ANN, on the other hand, must learn these patterns through multiple layers and weights, and without convolution it struggles to detect spatial structure in images. As a result the ANN becomes overwhelmed by the high dimensional input, while Logistic Regression remains simpler and more stable, allowing it to perform slightly better on this type of data.

Afterwards, we extracted the feature vectors from the CNN. These vectors were then reduced to 2 dimensions using PCA. The reduced feature space was used to apply k-means clustering.

The key idea here is to inspect whether classes with similar visual properties are grouped together. This allows us to explore potential semantic relationships between labels that were originally separated in the directory structure.

![figure 4](.\figure4.png)
**figure 4**

To determine the number of clusters to use with the clustering method, we used the elbow method and the silhouette method, to estimate how many clusters are appropriate for the feature vectors extracted from the CNN. The elbow method measures how the within cluster inertia decreases as the number of clusters increases, and it looks for a point where improvements slow down. The silhouette method evaluates how well samples fit within their assigned clusters compared to other clusters. These two perspectives help avoid choosing a number of clusters that is either too small or unnecessarily large. The meaning of this section depends on the shapes of the generated plots and the silhouette scores printed by the code.

![figure 5](.\figure5.jpeg)
**figure 5**
![figure 6](.\figure6.jpeg)
**figure 6**

We first applied the elbow method, but as shown above from the graph displaying the curve to apply the elbow method in figure 5, the exact position of the elbow is not clear. Therefore we tried applying the silhouette method, and the optimal number of clusters we found was k=2. 

However, our goal was to grasp a fair amount of semantic relationships given that we have over 200 different objects. For this reason, we decided to go back to considering the elbow method a choosing different number of clusters and graphically looking at how coherent, in terms of the circular shape, each cluster was. Through trail and error we found that k=5 was the best number of clusters. 

**From clusters to categories**

For each cluster, the code looks at which classes appear inside it based on whether that class has the highest number of images within that cluster compared to the other ones. By assigning each class to the cluster where it is most represented, the code builds a list of classes for every cluster. This helps to interpret the clusters not just as numbers, but as groups of semantically related image classes.

To understand the relationship between classes within each cluster, we manually found the main theme of each cluster. Moreover, for each cluster, we have created micro-themes. All of this was done manually by looking at the semantic relationship of the classes. As an example, as illustrated by the figure 7 below, we can see the title represents the theme of cluster 0, where natural, animals and diverse objects is the theme of the cluster and we can find examples of micro-themes such as animal and wildlife, which is the prominent micro-theme within this category. To clarify, as an example, the animal and wildlife micro-theme was manually created by looking at all the semantic fields related to animals within that cluster. For instance in the latter micro-theme leopards, zebra, porcupine and elk were just a few examples that were included.

![figure 7](.\figure7.jpeg)
**figure 7**

### Conclusions
The goals of this project of building classification models and uncovering semantic relationships between classes were achieved. Unfortunately, due to time limitations, the metrics have not reached the best results. However, we were still able to interestingly notice the strengths and weaknesses of the models. Moreover, it was extremely engaging and interesting to see that there were semantic relationships from our clustering analysis such as the animals example we made, despite the low raw accuracy of our models.

More specifically, in this project we explored the dataset, balanced it, trained multiple models, evaluated their performances and conducted clustering analysis on learned features. Our main finding is that CNN outperforms ANN and logistic regression, confirming the advantage of spatial feature extraction when working with images. The semantic analysis further showed that certain classes naturally grouped together based on common features.

Finally, the project leaves some opened questions. Not all clusters were perfectly interpretable, indicating that PCA with KMeans may not fully comprehend the structure of the dataset. Another limitation is related to class imbalance, which was addressed through data augmentation, but could still influence model behaviour and training and validation curves comparisons. Possible future improvements include trying different clustering algorithms and expanding the CNN architecture, increasing epochs. These approaches may provide deeper and more accurate insight into the semantic organisation of classes.