## **Music-Recommender-System**

<img src="https://github.com/EddieAmaitum/Music-Recommender-System/blob/main/Music%20Recommender%20System%20photo.jpg" alt="Music Recommender System photo" width="100%">

### **Why this is important?**

#### **The context:**

With the advent of technology, societies have become more efficient with their lives. At the same time, however, individual human lives have also become more fast-paced and distracted, leaving little time to explore artistic pursuits. Also, technology has made significant advancements in the ability to coexist with art and general entertainment. It has in fact made it easier for humans with a shortage of time to find and consume good content.

Almost every internet-based company's revenue relies on the time consumers spend on it's platform. These companies need to be able to figure out what kind of content is needed in order to increase customer time spent and make their experience better. Therefore, one of the key challenges for these companies is figuring out what kind of content their customers are most likely to consume.

Also most people enjoy listening to music and the challenge of recommending music to a user is easy to understand for a non technical audience.

#### **The objectives:**

The objective of this project is **to build a recommendation system to predict the top_n songs for a user based on the likelihood of listening to those songs.**

This project showcases my ability to rapidly learn and develop machine learning solutions while laying the foundation for building a robust production-grade tool that demonstrates the complete end-to-end machine learning process. In the future, I aim to deploy an interactive tool as a demonstration for prospective employers.

### **Data Dictionary**

The core data is the Taste Profile Subset released by the Echo Nest as part of the Million Song Dataset. There are two files in this dataset. The first file contains the details about the song id, titles, release, artist name, and the year of release. The second file contains the user id, song id, and the play count of users.

#### **song_data**
song_id - A unique id given to every song

title - Title of the song

Release - Name of the released album

Artist_name - Name of the artist

year - Year of release

#### **count_data**
user _id - A unique id given to the user

song_id - A unique id given to the song

play_count - Number of times the song was played

#### **Data Source**
[Million Song Dataset ](http://millionsongdataset.com/)

#### Why this dataset?

* It is freely available to the public.
* It is a large enough dataset for the purpose of this project.

### **Approach**
#### *Please refer to the python notebook for a detailed eplanation of the project and code used*
  
* Loading and understanding the data

* Data cleaning and feature engineering, some steps taken include:

  * I combined the datasets to create a final dataset for our analysis
  * For easier analysis I encoded user_id and song_id columns
  * I filtered the data such that the data for analysis contains users who have listened to a good count of songs
  * I also filtered the data for songs that have been listened to a good number of times
    
* Exploratory Data Analysis

* I built recommendation systems using 6 different algorithms:

  * Rank/Popularity - based recommendation system
  * User-User similarity-based collaborative filtering
  * Item-Item similarity-based collaborative filtering
  * Model based collaborative filtering / Matrix factorization
  * Clustering - based recommendation system
  * Content based recommendation system
    
* To demonstrate clustering-based recommendation systems, the surprise library was used.

* Grid search cross-validation was used to tune hyperparameters and improve model perfomance.

* I used RMSE, precision@k, recall@k and F_1 score to evaluate model perfomance.

* In the future I hope to improve the performance of these models using hyperparameter tuning.
