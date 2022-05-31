# Project 3 - Subreddit Classification (APIs & NLP)


## Introduction:

***Reddit*** is a social platform where members post content that is curated based on topics and promoted through voting.
Reddit is composed of hundreds of sub-communities, known as subreddits. In this project, i choose two subreddits, namely Nutrition and Keto.

***Nutrition*** A subreddit for the discussion of nutrition science.  Macronutrients, micronutrients, vitamins, diets, and nutrition news are among the many topics discussed. 

***Keto*** The Ketogenic Diet is a low carbohydrate method of eating. Place to share thoughts, ideas, benefits, and experiences around eating within a Ketogenic lifestyle. 

## Directory Structure
<details>
  <summary>Expand here</summary>

```
Project 3 - Subreddit Classification (APIs and NLP)
|__ codes
|   |__ 01_Data_Prep_reddit-via-pushShift.ipynb   
|   |__ 02_EDA_Subreddits-NutriKeto.ipynb   
|   |__ 03_Model_Tuning.ipynb
|   |__ cv_analyse.py
|__ datasets
|   |__ subr_nutrition.csv      
|   |__ subr_keto.csv        
|   |__ combined_subr.csv        
|__ image
|   |__ man.jpg
|   |__ metric1.png
|   |__ metric2.png
|   |__ metrics_score.png    
|   |__ Overfitting.png
|__ presentation
|   |__ NLP_Classification_NutriKeto.pdf  
|__ README.md
|__ Requirement.txt
```
</details>

# The Data Science Process

## Problem Statement

- We want to help the moderators to automate the classification and this will benefit the maintenance and sanity of each subreddits by posting to the correct group.

- Can we accurately label if a post is more relevant for *Nutrition* or *Keto*?

## Data Collection

- We scrapped around 2000 post for each subreddits via Reddit's pushshift API . Since Keto is more active and have more members, it reached the first 2000 post on Jan 30, 2022. Nutrition’s reached 2000 post on Feb 14, 2022.

***Sources:***

[r/Nutrition](https://www.reddit.com/r/nutrition.json)
        
[r/Keto](https://www.reddit.com/r/keto.json)


## Cleanup and Pre-processing 
        
- Cleanup Author that are from AutoModerator 
- Remove duplicates in Title
        
- ***Tokenizing*** (lower case and removed punctuations) 
- ***Stemming*** (transforms a word into its root form, e.g. eating/eats to eat, losing/loses to lose, why to whi, does to doe)

*The selftext is mostly removed, so in this project we will focus more on the ***Title*** only.*


## Modeling and Tuning:
We built several Classification models (DecisionTreeClassifier, SVC, NaiveBayes(MutiNomial), RandomForestClassifier, CountVectorizer, AdaBoostClassifier,GradientBoostClassifier,  BaggingClassifier, TFIDFVectorizer) and tuned each models by specifiying hyperparameters via GridSearch.

<img  src="image/accuracy.png" width=500 height=300/>



GradientBoost Classifier and BaggingClassifier are the top 2 best models in terms of Score or Accuracy.


### Overfitting

Most of the models have overfitting problems. This is a limitation that has never been addressed by this project. Further tuning of hyperparameters might able to mitigate this problem.


<img  src="image/Overfitting.png" width=400 height=200/>


- ***DTREE, ADA, TVEC*** = Train Score is 10% difference from Test Score
- ALL OTHERS = Train Score is 15% (or more) difference from Test


### Metrics

Nutrition as the Positive Class (=1)
Keto as the Negative Class (=0)

- ***Sensitivity*** is a measure of the proportion of actual positive cases that got predicted as positive (or true positive). Sensitivity is also termed as Recall.
- ***Specificity*** is a measure of the proportion of actual negative cases that got predicted as negative (or true negative).
- ***Precision evaluates*** the fraction of correctly classified instances or samples among the ones classified as positives.
- ***F-score or F1 Score*** is a measure of a prediction's accuracy. It is calculated from the precision and recall/sensitivity of the test.


<img  src="image/Metric1.png" width=300 height=200/>
<img  src="image/Metric2.png" width=300 height=200/>


- ***GradientBoost Classifier*** wins Accuracy, Sensitivity and F1 Score
- ***Bagging Classifier*** wins Specificity and Precision



# Summary:
- To answer our ***Business Problem***: Yes, we have selected the best model that can classify posts from two different subreddits based on their title.

We selected  ***Bagging Classifier*** as our best  model for this classification project, though *GradientBoost* Classifier tops Accuracy and F1, it wasn’t that much difference. The major difference is in Sensitivity and Specificity, we want our negative class to be predicted more accurately, as Keto is strict diet and should not be classified under the Nutrition subreddit. Bagging Classifier is highest in Specificity (True Negative) rate  with 83.5% rate compared to GradientBoost Classifier with only 78.5%.


## Recommendation:
- Tuning of hyperparameters to overcome overfitting
- Include other features like self-text and probably sentiment analysis score might improve our metrics.
- Include images or videos in our anaysis for more accurate prediction (which requires more knowledge on different ML domains)


