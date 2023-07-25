# ZomatoRestaurantReviewAnalysis
This project aims to perform sentiment analysis on Zomato restaurant reviews using Natural Language Processing (NLP) techniques. We will classify the reviews into positive and negative sentiments and build a Naive Bayes classifier to accomplish this task. The dataset used for this analysis can be found in the file Zomato Review Kaggle.tsv.
# Requirements
Make sure you have the following libraries installed:<br>

* numpy<br>
* matplotlib<br>
* pandas<br>
* nltk<br>
* scikit-learn<br>

You can install the required libraries using the following command:<br>
**`pip install numpy matplotlib pandas nltk scikit-learn`**<br>
# Dataset
The dataset contains restaurant reviews from Zomato. We will perform text cleaning and preprocessing to convert the text data into a format suitable for training the model.<br>

# Data Preprocessing
We perform the following steps to clean and preprocess the data:<br>

* Remove special characters and keep only alphabets.<br>
* Convert all text to lowercase.<br>
* Tokenize the text into words.<br>
* Remove stopwords (commonly used words that do not add much meaning).<br>
* Perform stemming to reduce words to their base form.<br>
* Bag of Words Representation<br>
* We use the bag of words representation to convert the preprocessed text into numerical features. The CountVectorizer class from scikit-learn is used to create a bag of words with a maximum of 1500 features.<br>

# Model Training
We use the Naive Bayes classifier (GaussianNB) to train our sentiment analysis model. The data is split into training and testing sets using an 80-20 split.<br>

# Prediction and Evaluation
We make predictions on the test set and evaluate the performance of our model using a confusion matrix and accuracy score.<br>

# Usage
To use this project, follow these steps:<br>

* Install the required libraries as mentioned in the "Requirements" section.<br>

* Clone this repository or download the project files.<br>

* Make sure the dataset file Zomato Review Kaggle.tsv is placed in the same directory as the Python script.<br>

* Run the Python script<br>

# Conclusion

After performing sentiment analysis on the Zomato restaurant reviews using the Naive Bayes classifier, we achieved an accuracy of approximately 71%. This means our model correctly predicted the sentiment (positive or negative) of the reviews in 71% of cases. The accuracy score can be further improved by exploring more advanced NLP methods and experimenting with different machine learning algorithms.<br>

This project demonstrates how to use NLP techniques for sentiment analysis on textual data, providing a valuable resource for anyone interested in exploring similar tasks or working with restaurant reviews.<br>

Feel free to use and modify this code for your own projects or datasets. Happy coding!
