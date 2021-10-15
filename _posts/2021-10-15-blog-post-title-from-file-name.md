## Udacity Data Scientist Nanodegree: Project 1

Project 1 of the Udacity Data Scientist Nanodegree program involves a CRISP-DM analysis of NYC Real Estate Data I found on Kaggle.com that involves sales of properties during a one-year period across the five boroughs of New York City. The source for this data can be found here: https://www.kaggle.com/new-york-city/nyc-property-sales.

I will attempt to answer the following 3 questions:

1. Does Manhattan sales trend differently than other boroughs, and if so, how?
2. Are older buildings selling better than newer buildings based on location?
3. Are bigger spaces more likely to sell in any particular neighborhood?

---

### As per the CRISP-DM or Cross-Industry Standard Process for Data Mining, I followed these 6 steps:

#### Step 1: Business Understanding
I needed to make sure I knew enought about real estate and the factors that may or may not affect sales as well as a general understanding of the boroughs or sections that make up the city of New York as well as some of the neighborhoods within those boroughs. Typically people not familiar with New York City know more about the borough of Manhattan from movies and TV, but have very little knowledge of the surrounding boroughs. Perhaps they know more about Brooklyn more recently, but probably have very little awareness of boroughs like the Bronx, Queens, or Staten Island.

#### Step 2: Data Understanding
In order to understand the dataset I was looking at some quick coding in Python using the Pandas package allowed me to get a high level overview. Reading the CSV file I downloaded from Kaggle I uploaded it to a DataFrame and ran a describe function to get a high level overview of the data as well as a histogram to see the level of data wrangling and clean up I might have to do, including possible dropping or imputing missing data.

#### Some Python Code

```python
# Load the CSV file inot a Pandas DataFrame where I can run some functions to tell me more about the data I was looking at
df = pd.read_csv('./nyc-rolling-sales.csv')
df.describe()

# Only 'object' categorical columns
cat_df = df.select_dtypes(include=['object']).copy()
cat_df
```

#### Step 3: Data Preparation
TODO

#### Step 4: Modeling
TODO

#### Step 5: Evaluation
TODO

#### Step 6: Deployment
TODO
