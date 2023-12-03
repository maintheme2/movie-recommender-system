# Introduction

A recommender system is a type of information filtering system that suggests items or content to users based on their interests, preferences, or past behavior. This project's goal is to employ a user-based collaborative filtering approach to suggest movies to a user. For this project a data set of 100,000 movie ratings provided by MovieLens, consisting of 943 users and 1682 movies was used. To suggest movies to the user, this dataset was utilized to identify people with comparable movie ratings.

# Data analysis 

The main dataset: u.data, consists of 100000 rows and 4 columns: user_id, item_id, rating and timestamp.
<p align="center">
<img src="figures/udata.png" width="250">
</p>

Another dataset, u.item, along with the rest of the data, provides a very important mapping: movie_id to movie_title.
<p align="center">
<img src="figures/uitem.png" width="600">
</p>

In the process of analyzing movie ratings given by users, it was found that there are 18 cases when the same user rated the same movie several times. Therefore we need to average such ratings and remove repeated evaluations.

There are several distributions of the data:
<p align="center">
<img src="figures/movie_distr.png" width="600">
</p>

<p align="center">
<img src="figures/movie_rated.png" width="600">
</p>

From the distribution of the users age we can observe that people from 20 to 30 years old rate films more often than others. Perhaps we will see that a particular occupation will be appearing more frequently than others.
<p align="center">
<img src="figures/age_distr.png" width="600">
</p>

<p align="center">
<img src="figures/gender_distr.png" width="600">
</p>

<p align="center">
<img src="figures/occupation_distr.png" width="600">
</p>

# Model implementation



## Model Advantages and Disadvantages

## Training Process


## Evaluation

# Results
