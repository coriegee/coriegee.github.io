---
layout: inner
title: Analyzing Podcast Title Keywords vs View Counts
permalink: /project_2/
---
<html>
<head>
  <style>
    h1 {
      font-size: 48px;
    }
  </style>
</head>
<body>
	<br>
	<br>
	<br>
	<div align="center">
	  <h1>Analyzing Podcast Title Keywords vs View Counts</h1>
	</div>
	<br>
	<br>
</body>
</html>

[View the full project on GitHub](https://github.com/coriegee/DOAC-podcast-NLP)

## Project Overview
In this project, we delve into the world of podcasts and explore the correlation between episode title keywords and view counts. Specifically, we examine the "Diary of a CEO" (DOAC) podcast, seeking to uncover insights that can guide content creation decisions, increase engagement, and optimize viewership. By analyzing the keywords in episode titles and their relationship with view counts, the aim is to provide actionable recommendations for enhancing the podcast's performance.  
<br>

## Problem Statement
There are many variables that are known to affect podcast episode view count, a few examples would be the notoriety of the host and guest speaker, the episode release datetime and the episode duration. We are going to focus on just one, title keywords. The main challenge in this project is to identify which keywords in episode titles have a significant impact on episode view counts. By understanding these correlations, we can provide insights to podcast hosts on crafting titles and choosing topics that resonate with their audience and maximize viewer engagement.  
<br>

## Data Collection and Pre-processing
The DOAC podcast episodes feature on most of the major platforms, Youtube was chosen to scrape podcast data from since most platforms do not show view count to the public. We start by scraping the @TheDiaryOfACEO channel and saving the data to a spreadsheet. The dataset contains `Title`, `URL`, and `Views` for 215 episodes uploaded by the channel over the last 2 years.  
{% highlight python %}
# Load the spreadsheet data into a pandas DataFrame
data = pd.read_excel('DOAC_youtube_dataset.xlsx')
{% endhighlight %}  
![DOAC Dataset Header](/img/posts/DOAC_dataset_head.png)  
<br>

To prepare the episode titles for analysis, we preprocess the text by converting it to lowercase, removing punctuation, removing episode numbers and human names using the spaCy library. This leaves us with 807 words. Title character length and title word count is also calculated for data exploration.  
{% highlight python %}
def preprocess_text(text):
    '''Prepare the title text for TF-IDF vectorizer'''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation)).replace('“', '').replace('”', '')

    # load spacy model
    nlp = spacy.load("en_core_web_sm")

    # Remove people's names
    doc = nlp(text)
    for name in [ent for ent in doc.ents]:
        if name.label_ == "PERSON":
            text = text.replace(str(name), '')

    # Remove less common names which aren't removed by the spacy model
    for remove_name in [' gawdat ', ' mo ', ' sinek ']:
        text = text.replace(remove_name, '')

    # Remove episode number using regex
    text = re.sub(r'e\d+', '', text)

    return text

# Apply preprocessing function to the 'Title' column
data['Preprocessed_titles'] = data['Title'].progress_apply(preprocess_text)

# Put the titles in a list
corpus = data['Preprocessed_titles'].tolist()
{% endhighlight %}  
![DOAC Processed Dataset Header](/img/posts/DOAC_processed_dataset_head.png)  
<br>

## Exploratory Data Analysis (EDA)
Before diving into the analysis, we explore the dataset's key statistics and visualize important aspects. We calculate basic statistics for the view count, title character length, and title word count. Some key things to note:
* The range of episode views are from 18,223 to 6,984,529 views. With an average of 747,133 views per episode. The skewness value of 3.287 indicates that the distribution of the data is right-skewed, i.e. the majority of the data points are concentrated on the left side of the distribution.
* The range of title word count are from 7 to 20. With an average of 13 words per episode title
![Views Histogram](/img/posts/DOAC_histogram_views.png)  
<br>

## Methodology and Algorithms
### TF-IDF
We employ the Term Frequency-Inverse Document Frequency (TF-IDF) method to analyze keyword importance. It combines how often a word appears in a document (TF) with how unique it is across all documents (IDF) to highlight words that are both frequent in a document and distinctive to that document collection. In this case, we will be calculating how frequent the word appears in an episode title and how unique it is across all titles. The formula for TF-IDF is below:

_TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)_

_IDF(t, D) = log2(Total number of documents D / Number of documents containing term t)_

_TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)_  
<br>

We will use the TfidfVectorizer from scikit-learn to compute TF-IDF scores.   

{% highlight python %}
# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the titles using the TF-IDF vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Convert the TF-IDF matrix to a DataFrame for easier analysis
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Calculate the average TF-IDF score for each word across all titles
average_tfidf_scores = tfidf_df.mean()

# Sort the words by average TF-IDF score in descending order
sorted_words = average_tfidf_scores.sort_values(ascending=False)

# Display the top N words with the highest average TF-IDF scores
top_n = 20 
top_words = sorted_words.head(top_n)
{% endhighlight %}  
<br>

This produced a sorted list of words by TF-IDF scores. The top TF-IDF scoring keywords are 'life', 'founder' and 'world', meaning they are both frequent in titles and distinctive to the title collection. This list doesn't really give us any insightful data to act upon just yet, so next we find the correlation between keyword TF-IDF and view count.  
![Top TF-IDF Scores](/img/posts/top_tf-idf_scores.png)  
<br>

Taking a look at the distribution of TF-IDF score for an example word 'founder', we observe that the distribution is right-skewed with the majority of TF-IDF score values are zero, since the word doesn't appear in most titles.  
![TF-IDF Histogram](/img/posts/DOAC_histogram_tf-idf.png)  
<br>

### Correlation Analysis
When examining the relationship between two continuous variables like episode keywords' TF-IDF scores and view counts, correlation analysis is a natural approach. However, since our data involves non-normal distributions and potential outliers, traditional correlation measures like Pearson's correlation coefficient may not be the best fit.  

Instead, we choose to employ Spearman's rank correlation coefficient. Spearman's correlation assesses the strength and direction of the monotonic relationship between two variables, making it less sensitive to the specific distribution of data points and better suited for our context. Additionally, Spearman's coefficient is more robust in the presence of outliers, as it relies on ranking the data rather than the raw values. This makes it a suitable choice when analyzing correlations in our dataset where keywords' TF-IDF scores and view counts might not follow a linear relationship.  

Spearman's rank correlation coefficient is calculated as:

_rho = 1 - 6 * sum(d_i^2) / (n * (n^2 - 1))_

where _d_i_ represents the difference in ranks between the two variables for each observation, and _n_ is the number of observations.  

{% highlight python %}
# Calculate Spearman's rank correlation coefficient and p-value between each keyword and view counts
correlation_results = {}

for keyword in tfidf_df.columns:
    keyword_column = tfidf_df[keyword]
    correlation_coefficient, p_value = spearmanr(keyword_column, data['Views'])
    correlation_results[keyword] = (correlation_coefficient, p_value)

# Sort the words by correlation in descending order
sorted_words = sorted(correlation_results.items(), key=lambda x: x[1][0], reverse=True)
{% endhighlight %}  
<br>

## Results and Interpretation
One way to visualize the results is through horizontal bar charts showing the top positively and negatively correlated keywords with view counts.  
![Top Keyword Correlation Coefficients](/img/posts/top_keyword_correlation_coeffic.png)
![Bottom Keyword Correlation Coefficients](/img/posts/bottom_keyword_correlation_coeffic.png)  
<br>

We won't interperet these results just yet. We first applied the commonly used significance level of p_value < 0.05 to indicate if the observed correlation coefficients are statistically significant. In other words, it is highly unlikely that such a correlation coefficient could have occurred due to random chance alone.  
{% highlight python %}
# Create a DataFrame, calculate if p < 0.05 and then filter out statistically insignificant keywords
word_corr_df = pd.DataFrame([[keyword, score_tup[0], score_tup[1]] for keyword, score_tup in sorted_words], columns=['word', 'coefficient', 'p_value'])
word_corr_df['statistically_significant'] = word_corr_df['p_value'].apply(lambda x: x < 0.05)
word_corr_df = word_corr_df[word_corr_df['statistically_significant'] == True]
{% endhighlight %}  
![Correlation DataFrame](/img/posts/correlation_dataframe.png)  
<br>

In this case, we are left with only 22 keywords that are considered statistically significant in correlation to views. The majority of words have only a small sample size across the entire dataset and thus there is possibly not enough data points to detect significant effects.  
<br>

All the correlation coefficients are between 0.3 and -0.3, meaning the strongest correlations are still considered weak. There are of course many variables that will affect the view count of episodes so to find some weak correlations here is still somewhat interesting and potentially useful.
These 22 keywords are visualised here:  
![Significant Keywords with Strong Correlation](/img/posts/significant_keywords_with_strong_corr.png)  
<br>

**There are some interesting findings here as well as some expected results**

Positive Correlations:
* "man" was the title word that had the highest positive correlation with episode view count, suggesting that the audience is more inclined to watch an episode if the guest speaker is male, possibly due to a larger male audience (just a hypothesis without access to the data to validate this).
* Lots of words within the "Health and Wellbeing" category ("weight", "loss", "calories", "sugar", "sleep") that have a positive correlation to views, suggesting the audience is more likely to watch Health and Wellbeing topics than anything else.
* The nouns "doctor", "expert" and "scientist" suggesting viewers prefer to watch episodes with guests that are considered to be very knowledgeable about or skilful in a particular area. Perhaps this is expected, but it is valuable to know that these words should be included in the title if applicable.

Negative Correlations:
* Interestingly, "founder" has the most negative correlation to view count at -0.29. Suggesting that the viewers are not that interested in conversations with guest founders and are less likely to watch an episode when this word is mentioned.
* Similarly, "billion" and "dollar" are negatively correlated to view count, indicating the audience of DOAC are suprisingly not as interested in watching episodes about making money or businesses as they are about losing weight and improving health.  
<br>

### Validation
It will help us to understand and validate this difference between the most positively and negatively correlated words "man" and "founder" in terms of view count if we visualise them together in a scatter plot.
![Scatterplot](/img/posts/scatterplot.png)
Each data point is a single title. By removing the titles with zero-count occurences (TF-IDF score of 0) we can quite easily see here that there is indeed a significant difference between the red and blue clusters. When the word "man" is used in the title the number of episode views is increased relative to when the word "founder" is present in the title, validating our Spearman's rank correlation coefficient results above.  
<br>

## Conclusion
We've successfully analyzed the correlation between episode title keywords and view counts for the "Diary of a CEO" podcast. We've highlighted keywords that have a significant impact on viewer engagement and provided actionable recommendations for improving content creation. While there are limitations to the analysis, such as weak correlations and the influence of other factors, the insights gained contribute to informed decision-making in podcast production.

By applying natural language processing techniques and correlation analysis, we've empowered podcast hosts to enhance their reach, resonate with their audience, and ultimately create more impactful content. This project showcases the power of data science in uncovering hidden patterns and guiding strategic decisions in the podcasting realm.
<br>

[View the full project on GitHub](https://github.com/coriegee/DOAC-podcast-NLP)