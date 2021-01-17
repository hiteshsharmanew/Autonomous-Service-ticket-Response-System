# Language/Tools Used

Tools: Python 3. Packages: Matplotlib ,NLTK ,SciPy ,pandas ,scikit-learn , Mysql database , Outlook

# Business Case Scenario

Organisations all over the globe spend a considerable amount of resources towards organising/maintaining a service ticket system . Conventionally it is usually handled by a service team which handles internal problems locally and a central team governing all the region specific resources. Now naturally each complaint caters to a different need , which takes a great time wasted in proper allocation of the problem to the concerned team and thus increases the response time of service team to come up with solutions.
The method of contacting the team could be anything from a phone call , emails , chat on a dedicated server etc.

The consisted of builiding a software for the client which would be able to reduce this unnecessary manual effort and decrease the response time considerably. This also increases the productivity of the service teams and reduces team expenses
Our solution paticularly handles the problems coming through the email channel and then it was integrated with other capabilities like handling channels such as phone calls , chatbots.

# Data Set
The data set contains 34K actual mails received by the service which have relevant columns such as Incident number (Unique identifier) , Request date ,  Summary , Description of the mails , Assigned /group etc . 
Objective  : Multi- classification (Categorisation of emails) .
Train/Test split is a conventional 80-20 split.

# EDA (Data visualisation)
The graphs demonstrate the distribution of various emails amongst various features . Please head to visualisation folder for further details. 

# Solution Design 
Natural Language Processing has long been used to build text classification system to categorize news articles, analyzing app or game reviews using topic modeling and text summarization, and clustering popular movie synopses and analyzing the sentiment of movie reviews.

This project utilizes the same concepts in order to categorize the emails into 9 different categories to build a machine learning model at the back end , with a bot continous monitoring emails . After processing the mails the model assigns the emails to the concerned team by uploading the mail to a database and appropriate mails are sent accordingly.

# Pre - Processing 
The code discusses the various techniques implemented to clean the data . Pretty standard in textual recognition (stemming , normalisation , stop words , contractions etc ) . Please visit the Pre processing folder for more detail.

# Machine Learning models used
Bag of words 
Two different models have been tested/validated (SVM , Naïve Bayes).
Max accuracy : 89% , Input data : text based (contextual recognition)

