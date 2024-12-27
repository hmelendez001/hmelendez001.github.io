## Udacity Data Scientist Nanodegree: Project 4 Final Capstone Project

As part of the udacity.com Data Scientist nanodegree, this is project four and the final project of the program, otherwise known as our Capstone Project. This project involves building a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. The inputs will contain multiple metrics, such as opening price (Open), highest price the stock traded at (High), how many stocks were traded (Volume) and closing price adjusted for stock splits and dividends (Adjusted Close); we only need to predict the Adjusted Close price.

Investment firms, hedge funds, and even individuals or day traders have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process. According to an article by Mark Kolakowski ["How Robots Rule the Stock Market (SPX, DJIA)"](https://www.investopedia.com/news/how-robots-rule-stock-market-spx-djia):

> only 10% of trading volume now comes from human discretionary investors, per data from JPMorgan Chase & Co. (JPM) cited by Bloomberg

![Crypto Trading Robot Image](https://www.influencive.com/wp-content/uploads/2021/04/CryptoTrading-Bots-758x511.png)
##### Source: "29 Best Crypto Trading Bots on the Market" by FRED at https://www.influencive.com/29-best-crypto-trading-bots-on-the-market/

---
### Problem Statement
For this project, I built a stock price predictor that reads historical daily trading data over a certain date range as input from the Yahoo! Finance API, and outputs projected closing price predictions for given future query dates. The inputs contain multiple metrics for any given day:

- Open: stock opening price, 
- High: highest price the stock traded, 
- Low: highest price the stock traded, 
- Volume: how many stocks were bought and sold,
- Close: the closing unadjusted price of the stock, and
- Adjusted Close: the closing price adjusted for stock splits and dividends

For historical test data we only need to predict the Adjusted Close price, however, for future dates we need to predict the inputs as well.

![Trading Floor Broker](https://image.cnbcfm.com/api/v1/image/106427063-1583430452588gettyimages-1205293048.jpeg?v=1583430647&w=740&h=416)
##### Source: "SPACs break 2020 record in just 3 months, but the red-hot industry faces challenges ahead" by Yun Li at https://www.cnbc.com/2021/03/19/spacs-break-2020-record-in-just-3-months.html
###### Traders work during the opening bell at the New York Stock Exchange (NYSE) on March 5, 2020 at Wall Street in New York City. (Photo by Johannes EISELE / AFP) (Photo by JOHANNES EISELE/AFP via Getty Images)
###### Photo by Johannes Eisele | AFP | Getty Images

---
### Data Mining
For my primary market data source I considered a few suggestions like [Yahoo! Finance API](https://www.yahoofinanceapi.com), [Bloomberg API](https://www.bloomberg.com/professional/support/api-library/), and [NASDAQ Financial API](https://data.nasdaq.com/tools/api) as per our [Udacity Project Description](https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub). My Wall Street background immediately drew me to Bloomberg as the obvious choice, but there are pricing and licensing limitations that come with Bloomberg. So, I decided on Yahoo! Finance mainly because there was no need to sign up and I found a lot of online references for Python rather easily. It offers an API that is easy to use and well documented. So that is what ultimately drove my decision. I downloaded the data with the following code and saved the csv file for the analysis:

```
import yfinance as yf  
import matplotlib.pyplot as plt

data = yf.download('BTC-USD','2021-01-01','2021-12-31')
data = data.reset_index()

data["Close"].plot()
data.to_csv("data/bitcoin_stock_2021.csv", index=False)

plt.show()

```
##### Caveat Emptor and Disclaimer
APIs, and especially Financial Markets APIs like Yahoo! Finance, have limits on how much data you can pull and how frequently. I will minimize the number of times I run code like the one above to avoid hit these limits, and heaven forbid, get rate limited or blacklisted. I will show how we can cache requests as much as possible so that I am not hitting the API as frequently. Since this is an academic exercise, I am not concerned with real-time accuracy just data to run the models against, so data is cached on a per day basis: meaning, I call the API once per symbol and date range, and store the results in a database to minimize the number of times we call the API.

**My disclaimer: I am not a certified financial planner nor a financial advisor nor a certified financial analyst nor an economist nor a CPA. These predictions and recommendations are only for academic purposes and not an official recommendation from Udacity nor from my employer RSM in any way.**

![Yahoo! Finance API](https://algotrading101.com/learn/wp-content/uploads/2020/05/yahoo_homepage-2-1024x545.png)
##### Source: "Yahoo Finance API – A Complete Guide" by Greg Bland at https://algotrading101.com/learn/yahoo-finance-api-guide/

---
#### Other Financial Indicators
As part of the data wrangling, I also wanted to bring in other factors I feel have an influence on stock prices in general. Some of this data like the Department of Labor Weekly Unemployment Numbers will require some wrangling to extrapolate a daily number since it is published weekly, and some will be easy to both acquire and merge with our stock price and volume data because it is market data that trades with a stock ticker. So, I also pulled that down and saved to the data folder like with the stock price data.

1. Weekly Unemployment Numbers from https://oui.doleta.gov/unemploy/claims.asp: unemployment numbers typically drive market numbers. I can remember every Thursday when working on Wall Street what a big deal it was when these numbers came out from the US Department of Labor<br>
2. CPI or Consumer Price Index: how well is the consumer market spending is a typical bell weather of consumer confidence<br>
3. DJIA or Dow Jones Industrial Average Index: an overall market indicator<br>
4. Google Global Mobility Report from https://www.google.com/covid19/mobility: how much are people getting around despite of the COVID-19 global pandemic, filtered down to just US numbers<br>
5. SP&500 Index: ^GSPC ticker: additional market indicator<br>
6. NASDAQ Composite Index: ^IXIC ticker: Over the Counter market indicator<br>

![DJIA: Dow Jones Industrial Average](https://cdn.corporatefinanceinstitute.com/assets/Dow-Jones-Industrial-Average-1.jpeg)
##### Source: "What is the Dow Jones Industrial Average (DJIA)?" at https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/dow-jones-industrial-average-djia/

---
#### Data Preprocessing
The Weekly Unemployment Numbers came as an Excel file, so I opened it in Excel and saved it as a CSV for our convenience: data/DOL_Weekly_Unemployment_US_2021.csv.

For the Google Global Mobility Report, I read in the global report into a Pandas DataFrame because this file was too big to simply open in Excel like I did for the Weekly Unemployment Numbers.

```
import pandas as pd
import numpy as np

df = pd.read_csv('data/Global_Mobility_Report.csv')

# Show df to get an idea of the data
df.head()
```

Then I noticed it was as expected, but more than US national numbers: US state numbers. And more than just 2021. So, I took just the data I needed from this data frame and wrote it out for later use:

```
df_us = df[(df['country_region_code'] == 'US') & (df['date'] >= '2021-01-01')]
df_us.head()
#duplicates_count = df_us.duplicated(subset='date', keep='first').sum()
#assert duplicates_count == 0, "The Google duplicate count should be zero, but it is {}".format(duplicates_count)

# Oops! There are duplicates? No, it's just that Google breaks it down by state/region, so let's ignore that data
# Let's only focus on the US wide numbers. If 'sub_region_1' is not set then this is the US-wide number, what we want
df_us = df_us[~df_us['sub_region_1'].notnull()]
duplicates_count = df_us.duplicated(subset='date', keep='first').sum()
assert duplicates_count == 0, "The Google duplicate count should be zero, but it is {}".format(duplicates_count)
df_us.to_csv("data/Global_Mobility_Report_US_2021.csv", index=False)
```
![Google COVID-19 Community Mobility Report](https://www.google.com/covid19/mobility/static/image.png)
##### Source: "COVID-19 Community Mobility Reports" at https://www.google.com/covid19/mobility

---
For the others I kept the format as is in the data folder. So, for the analysis that leaves me the following 7 files to merge:

1. Weekly Unemployment Numbers: data/DOL_Weekly_Unemployment_US_2021.csv<br>
2. CPI or Consumer Price Index: data/cpi_index_2021.csv<br>
3. DJIA or Dow Jones Industrial Average Index: data/djia_index_2021<br>
4. Google Global Mobility Report: data/Global_Mobility_Report_US_2021.csv<br>
5. SP&500 Index: data/sp500_index_2021.csv<br>
6. NASDAQ Composite Index: data/nasdaq_composite_index_2021.csv<br>
7. Bitcoin USD Ticker Data: data/bitcoin_stock_2021.csv<br>

### Initial Findings
A correlation analysis using the Weekly Unemployment numbers revealed no correlation, at least for our USDB Bitcoin ticker, so we also checked against other equities (Tesla Motors and Texas Instruments, but same results. Perhaps the Weekly Unemployment has correlation to the Fixed Income market (bonds, loans, structured debt, etc.), but not the Equity names I analyzed. In fact, none of the economic indicators we anticipated to correlate to our stock prices correlated in a significant way.

![Market Indicator Correlation Analysis](https://user-images.githubusercontent.com/26253570/148664284-ca49412e-9437-49a9-b1a7-4ddaa155745e.png)

---
Okay, some positive correlation, but also some negative effects from the following: 

`a.` parks_percent_change_from_baseline: people going to the park or not apparently does not affect Bitcoin value<br>
`b.` residential_percent_change_from_baseline: interesting there is negative correlation since more people are working from home during the pandemic<br>
`c.` S&P Volume: Volume of blue-chip trading has no correlation<br>
`d.` NASDAQ Volume: Neither does volume of over-the-counter trading at the NASDAQ<br>

So, let's update our status table...

| No | Indicator | Data Source | Description | Status |
|----|-----------|-------------|-------------|--------|
| 1 | Weekly Unemployment | data/DOL_Weekly_Unemployment_US_2021.csv | I can remember every Thursday when working on Wall Street what a big deal it was when these numbers came out from the US Department of Labor | DONE, no correlation |
| 2 | CPI or Consumer Price Index | data/cpi_index_2021.csv | Indicates strength of consumer market spending, a typical bell weather of consumer confidence | DONE, correlation found |
| 3 | DJIA or Dow Jones Industrial Average Index | data/djia_index_2021.csv | Overall blue-chip market indicator | DONE, correlation found |
| 4 | Google Global Mobility Report | data/Global_Mobility_Report_US_2021.csv | How much are people getting around despite of the COVID-19 global pandemic, filtered down to just US numbers | DONE, some correlation found except for **points a and b** above |
| 5 | SP&500 Index | data/sp500_index_2021.csv | ^GSPC ticker: additional market indicator | DONE, some correlation found except for **point c** above |
| 6 | NASDAQ Composite Index | data/nasdaq_composite_index_2021.csv | ^IXIC ticker: Over-the-Counter market indicator | DONE, some correlation found except for **point d** above |

...and even after I re-ran my model with this newfound knowledge, I concluded that with less work, i.e., no economic indicator data, we can do a better job of predicting future stock value. Our linear regression model was a good start using just the historical data provided by Yahoo! Finance on the stock symbol.

![NASDAQ Composite Scores](https://static5.businessinsider.com/image/5e050b69855cc259ee177af4-2000/rtx2q58v.jpg)
##### Source: "The Nasdaq soars past 9,000 for the first time ever, fueled by Amazon's holiday sales boom" by Ben Winck at https://markets.businessinsider.com/news/commodities/nasdaq-composite-hits-9000-first-time-amazon-sales-santa-rally-2019-12 Reuters/Shannon Stapleton

---
### Implementation - The Stock Predictor
First, I created some helper functions and classes that would be needed for validating and pre-processing the inputs. I also included some unit testing with these just as a sanity check. This helped me make changes more confidently as well as refactor code as needed while being able to run some quick regression tests to make sure everything was working as expected:

`1.` Function `is_valid_range(from_date, to_date, can_be_in_the_future = False)`: Is the given date range a set of valid dates, by default they have to be in the past or historical dates, but for making future prediction the date range can be in the future<br>
`2.` Function `def get_from_cache(symbol, from_date, to_date)`: Check for symbol and validate ranges to see if we have cached them already. Even invalid symbol calls to the API were cached so that we would not call invalid symbols for the purpose of regression testing more than once per run<br>
`3.` Function `get_stock_history(symbol, from_date, to_date)`: This was the function that actually checked the cache and then called the Yahoo! Finance API to get historical data as needed<br>
`4.` Class `TrainedModel`: At first this was a class I created to encapsulate and cache both the model as well as associated parameters like training and test dates, etc., but then I realized I could use this abstract base class to switch in and out the type of underlying machine learning model in the future. This class defines the functions: `fit`, `predict`, and `evaluate_model`<br> 
`5.` Class `TrainedModelLinear`: Child class of `TrainedModel` which uses `LinearRegression` to implement the price predictions<br>
`6.` Function `GetTrainedModel(symbol, df_symbol, newFromDate, newToDate)`: Returns the concrete implementation `TrainedModel` object to use<br>
`7.` Function `are_valid_symbols(symbols)`: Determines if the given parameter is a valid list of stock symbol tickers and all the given symbols have corresponding trained models cached<br>
`8.` Function `evaluate_models(symbols)`: Iterates through a validated list of ticker symbols and calls the evaluate_model function for each corresponding `TrainedModel`<br>
`9.` Function `get_stock_histories(from_date, to_date, symbols, test_size, random_state)`: Iterates through a validated list of ticker symbols and calls `get_stock_history` for each then fits the corresponding `TrainedModel` object<br>
`10.` Function `get_stock_predictions(from_date, to_date, symbols, max_days_to_predict)`: Iterates through a validated list of ticker symbols and makes predictions for the given date range<br>

![Stock Market Deep Learning](https://miro.medium.com/v2/resize:fit:720/format:webp/1*oVsvog1FPoig-lG4RJzN-A.png)
##### Source: "Stock Market Prediction using Deep Learning" at https://kingsubham27.medium.com/stock-market-prediction-using-deep-learning-b71ae6fea740

---
### Test and Measure Performance: How Good Are the Predictions?
The `evaluate_model` function for class `TrainedModelLinear` predicts the accuracy of the model by taking the mean of the actual closing prices over the mean of the predicted closing prices as a percentage of accuracy, and because it is a linear model, I also calculate the R-square score. 

I trained the linear `TrainedModelLinear` for 2021 Tesla Motors (TSLA), Apple (AAPL), and Microsoft (MSFT). Then I trained the model for the first 9 months of the year 2021, and tested the last 3 months: 1 day, 7 days, 14 days, one month, and finally the last 3 months cumulative and saw the following results below.

![Model Evaluation Across: Telsa, Apple, and Microsoft](https://user-images.githubusercontent.com/26253570/148699017-5b1ef6f7-a5c5-40fa-8a34-ff7590e70126.png)

---
I still see promising accuracy, close to 100 or a little over, meaning our predictions were still on the conservative side or a little lower than the actual price for all 3 names. However, notice that Tesla Motors (TSLA) is probably the most volatile of the 3 Equity names as it has the highest error (MAE, MSE, and RMSE).

### Refinement and Potential Improvements
Although all 3 have a good linear fit, R-square scores relatively close to 1, and overall accuracy within 100, there is definite room for improvement here. Perhaps analysis of additional indicators that are specific to the stock in question to analyze the health of the underlying company ("P/E ratio or price-to-earnings ratio, P/B ratio or price-to-book ratio, liquidity ratios, debt ratios, return ratios, Margins, etc.") or even additional models beyond a linear fit or the basic technical analysis being done here. There are other forms of analysis that include NLP or Natural Language Processing where I can evaluate news, tweets, and social media posts associated to the company, otherwise known as Sentiment Analysis.

If we go with another model, notice that our code relies on a base class `TrainedModel` so that we could more easily swap out our `TrainedModelLinear` with say `TrainedModelLSTM` deep learning. From [Wikipedia: Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory):

> Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDSs (intrusion detection systems).

Many of the existing Stock Predictor models I found online employ LSTM because of the sequential nature of the historical data.

![LTSM Stock Predictor](https://miro.medium.com/max/1400/1*opzxrBna63YDbd8_pM5trw.png)
##### Source: "Stock price prediction using LSTM (Long Short-Term Memory)" by Thenuja Shanthacumaran at https://medium.com/analytics-vidhya/stock-price-prediction-using-lstm-long-short-term-memory-e8c125a853e4

Sidebar: There were headlines around Sentiment Analysis where bots picked up news of a fictional character dying on a Peloton bike. This prompted a sell off of the company stock prompting analysts to ask where the trading bots and algorithms fooled into confusing news of a fictional character's death as real news or where they actually smarter than that and knew it was a fictional character but anticipated how this would affect the actual company's image: [Peloton stock slumps after morbid product placement in "Sex and the City"](https://www.cbsnews.com/news/peloton-stock-death-by-peloton-just-like-that-mr-big)

> Shares of Peloton, the fitness equipment company, fell 11.3% Thursday — tumbling to a 19-month low — after a key character in HBO Max’s “Sex and the City” revival, “And Just Like That,” was shown dying of a heart attack after a 45-minute workout on one of the company’s exercise bikes.

![Peloton Stock Price Plunge](https://s.yimg.com/uu/api/res/1.2/5NmDbiZfrZcB0BLoR3zQ2A--~B/Zmk9ZmlsbDtweW9mZj0wO3c9NjQwO2g9MzYwO3NtPTE7YXBwaWQ9eXRhY2h5b24-/https://s.yimg.com/hd/cp-video-transcode/prod/2021-12/10/61b37cc424ffa931eb82cb23/61b38fcc78856f75e6071d4c_o_U_v5.jpg)
##### Source: "Mr. Big gives Peloton stock a heart attack — investors flee" by Brian Sozzi at https://finance.yahoo.com/news/mr-big-gives-peloton-stock-a-heart-attack-investors-flee-182640731.html

---
### Content Based Recommendations

I also wanted to make some stock recommendations based on content. Like for my previous Udacity project where I made recommendations based on IBM articles, but for our stock names. For example, you like Bitcoin, perhaps you might like these other Crypto currency names that trade in the same price range, or perhaps you are interested in other tech sector names? So I downloaded data from [SwingTradeBot.com](https://swingtradebot.com/equities?min_vol=1000&min_price=10.0&max_price=999999.0&adx_trend=&grade=&include_etfs=2&html_button=as_html) that gives me some characteristics across equity stocks so I can build a recommendation engine across factors.

But even though financial data is great for data science in that a lot of it is already numerical and great for modeling out of the gate, it is not so much for content recommendation. Like movies or IBM articles, numbers, prices, or trading volumes are not categories like genre, themes, or other related categories. Outside of the stock company industry or sector, for the numeric values I used the Pandas describe function to identify 25/50/75 percentiles to classify the numbers into binary columns or buckets, e.g. In25Pct_close_price, In50Pct_close_price, In75Pct_close_price, In100Pct_close_price, and we do this for volume, change in percent, days old, adx, ..., peg, eps, div_yield, and atr, etc.

![Stock Prediction Charts](https://m.foolcdn.com/media/dubs/images/Investing_dice_buy_sell.original.jpg)
##### Source: "50 Everyday Costs You're Overlooking That Are Adding Up" by Selena Maranjian at https://kingsubham27.medium.com/stock-market-prediction-using-deep-learning-b71ae6fea740

---
Wall Street analysts typically look at stocks within the same sector or industry, but to give my recommendation engine some novelty and potential serendipity, I built my recommendations to go across all sectors or industries and an option to only stick within similar sector/industry. I categorized the 13 unique sectors because the 113 unique industries are probably too many for now. Of the 13 there was one nan or NaN. I needed to account for this and simply created an 'Unknown' sector. So, this plus the 25/50/75 percentiles columns left me with a 98-column matrix. I then reduced this to just the binary value columns so that I could use a dot matrix product to produce a "similarity" score for any combination of stocks. 

![Stock Market Deep Learning](https://user-images.githubusercontent.com/26253570/148699756-a24c4961-ebaf-4437-b76e-d2f7f060476e.png)

I created a function called `make_content_recs(symbol, m=10, sort_by_sector_industry=False, df_cat_new=df_cat_new)`, by default for any given symbol it gives me the top 10 most similar stock symbols. I then ran the following tests: notice I ran Apple stock for both same and different sectors to see what kind of serendipitous stock recommendations we might get:

```
# test non-existent symbol
print('*** ERROR does not exist: ZZVZT', make_content_recs('ZZVZT'))
        
# test existing symbol
print('*** IBM:', make_content_recs('IBM'))
# both by any sector and by related sector to get different results
print('*** Apple any sector:', make_content_recs('AAPL'))
print('*** Apple SAME sector:', make_content_recs('AAPL', 10, True))
```
And I got the following results:

```
*** ERROR does not exist: ZZVZT ("*** Symbol 'ZZVZT' does not exist", [], [])
*** IBM: ('', ['AAPL', 'AMD', 'NVDA', 'BBIO', 'CCL', 'AMC', 'AVCT', 'BAC', 'AAL', 'BBD'], ['Apple Inc.', 'Advanced Micro Devices, Inc.', 'NVIDIA Corporation', 'BridgeBio Pharma, Inc.', 'Carnival Corporation', 'AMC Entertainment Holdings, Inc.', 'American Virtual Cloud Technologies, Inc.', 'Bank of America Corporation', 'American Airlines Group, Inc.', 'Banco Bradesco SA'])
*** Apple any sector: ('', ['AMD', 'NVDA', 'BBIO', 'CCL', 'AMC', 'AVCT', 'BAC', 'AAL', 'BBD', 'TSLA'], ['Advanced Micro Devices, Inc.', 'NVIDIA Corporation', 'BridgeBio Pharma, Inc.', 'Carnival Corporation', 'AMC Entertainment Holdings, Inc.', 'American Virtual Cloud Technologies, Inc.', 'Bank of America Corporation', 'American Airlines Group, Inc.', 'Banco Bradesco SA', 'Tesla Motors, Inc.'])
*** Apple SAME sector: ('', ['AMD', 'NVDA', 'FB', 'APPS', 'MSFT', 'ATVI', 'AEY', 'SQ', 'BB', 'AMAT'], ['Advanced Micro Devices, Inc.', 'NVIDIA Corporation', 'Facebook, Inc.', 'Digital Turbine, Inc.', 'Microsoft Corporation', 'Activision Blizzard, Inc', 'ADDvantage Technologies Group, Inc.', 'Block, Inc', 'BlackBerry Ltd', 'Applied Materials, Inc.'])
```

#### How I would improve the content-based recommendation system? 
Notice that even though I was looking at two technology stocks: IBM and Apple, the recommendation engine gave me some symbols or names not in the technology sector: a pharma BridgeBio Pharma, Inc., a leisure-and-travel stock: Carnival Corporation, an entertainment one: AMC, and even a bank: Bank of America Corporation. But how to measure and improve my recommendations? I could compare against what other online recommendation engines make. Sites like [seekingaplha.com](https://seekingalpha.com) will provide stock recommendations by sector but that comes at a premium. In fact, I would be hard pressed to find anything online that will give me these types of recommendations for free. At best, for a stock like Apple, I can find some online articles or sites (https://www.tipranks.com, https://finance.yahoo.com, or https://marketchameleon.com) that recommend similar symbols and I can see AMD, NVDA, TSLA, amongst others also appear in these types of recommendations. But this is just a small sample of data and not enough to tell us how precise my picks are. So, this evaluation is left as a potential future exercise, at best.

For the classic "cold start" problem with the content-recommendation engine I would probably fall back to top volume traded or sort by VWAP (Volume Weighted Average Price) to sort by best value. I created a second function `get_top_stocks(m=10, sector='', df_cat_new=df_cat_new)` for cold starts. Then I ran the following tests:

```
# make recommendations for a brand new user no stock picked yet...just return the top volume stocks
print ('Top 10:', get_top_stocks(10))

# Test a bad sector
print ('XYZ', get_top_stocks(10, 'XYZ'))

# Get top Healthcare stocks
print ('Healthcare', get_top_stocks(10, 'Healthcare'))
```
This gave me the following results:
```
Top 10: ('', ['AAPL', 'AMD', 'NVDA', 'BBIO', 'CCL', 'AMC', 'AVCT', 'BAC', 'AAL', 'BBD'], ['Apple Inc.', 'Advanced Micro Devices, Inc.', 'NVIDIA Corporation', 'BridgeBio Pharma, Inc.', 'Carnival Corporation', 'AMC Entertainment Holdings, Inc.', 'American Virtual Cloud Technologies, Inc.', 'Bank of America Corporation', 'American Airlines Group, Inc.', 'Banco Bradesco SA'])
XYZ ("Sector 'XYZ' does not exist, try one of ['Healthcare' 'Basic Materials' 'Other' 'Unknown' 'Consumer Defensive'\n 'Financial Services' 'Industrials' 'Technology' 'Consumer Cyclical'\n 'Real Estate' 'Communication Services' 'Energy' 'Utilities']", [], [])
Healthcare ('', ['BBIO', 'BFRI', 'NVAX', 'MRNA', 'ALLK', 'ARDX', 'AKBA', 'MDT', 'BSX', 'BMY'], ['BridgeBio Pharma, Inc.', 'Biofrontera Inc.', 'Novavax, Inc.', 'Moderna, Inc.', 'Allakos Inc.', 'Ardelyx, Inc.', 'Akebia Therapeutics, Inc.', 'Medtronic Inc.', 'Boston Scientific Corporation', 'Bristol-Myers Squibb Company'])
```

![Top 12 Tech Stocks for 2022](https://www.zenectwealth.com/wp-content/uploads/2021/12/stock-chart.jpg)
##### Source: "Best Tech Stocks To Buy Right Now? 5 To Watch by Amos C at https://www.zenectwealth.com/best-tech-stocks-to-buy-right-now-5-to-watch/

---
### Summary
I explored the Yahoo! Finance historical trading data including using the Yahoo! Finance API to get live historical data. We also loaded well known financial indicator data, did some analysis and clean-up of this data, and showed there was not enough significant correlation to our pricing models to warrant their use. So, although I disproved my initial assumption that this financial indicator would help me in predicting closing prices, this meant the model was a little easier to implement. I then created a Stock Predictor to predict Adjusted Closing prices based on previous day's closing data, and finally created a Content-Based Recommendation Engine to suggest similar Stocks based on most similar Stock attributes for a given Stock.

### Conclusion

#### Reflection
This was a very fun project for me. Being able to choose a subject matter that I both find interesting and familiar, yet looking at it with a Data Science lens, made this work fascinating to me. It was a great opportunity to apply everything I learned and showcase how it could be applied to a real-world problem and one that is close to home. The hardest part was to stop adding extra work. I probably could have kept working on this, thinking of new ways to model and test this. By far the best project in the Udacity program.

This being the Data Science final Capstone Project, I completed the nanodegree. Udacity even provided a "graduation ceremony" video at the end of the program, with [Sebastian Thrun](https://www.linkedin.com/in/sebastian-thrun-59a0b273/) as the "commencement speaker." He said before the self-driving car problem was solved, and everyone had all but given up on solving it, Google told him: tell us why technically it's not possible to do it. When he could not come back with a logical, technical reason why a self-driving car could not negotiate the public roads, he was renewed for the challenge to ultimately solve the problem. And the rest is [Google autonomous car history](https://www.ted.com/talks/sebastian_thrun_google_s_driverless_car?language=en).

![Google Self-Driving Car Launched in 2015](https://a.scpr.org/i/41a146393791de83a04a8e83fa197bb3/83369-full.jpg)
##### Source: "Google's new self-driving cars cruising Silicon Valley roads by Michael Liedtke at https://archive.kpcc.org/news/2015/06/25/52700/googles-new-self-driving-cars-cruising-silicon-val

Since I have the code basis for deploying a web application that will both showcase our Stock Predictor for Adjusted Closing price and our Content-Based Recommendation Engine, I went ahead and did that. This way I can reach a wider audience. So here is my final web application: [My Stock Predictor](https://helderstockpredictor.herokuapp.com). Other than the last week of January 2022 when the market went sideways the predictions have been very close with R² close to 1.00. But that is why this was purely an academic exercise. I'll leave the trading decisions to the experts. I'll continue with my data science now.

![Helder's Stock Predictor Web App](https://user-images.githubusercontent.com/26253570/150998912-7f894dfe-f9cc-4402-af6f-f4b6ec4ba6ec.png)

And of course, below is [my "diploma" for completing the Data Scientist Nanodegree Program](https://confirm.udacity.com/WKHPMPA2). Happy Wrangling!

![Helder's Nanodegree Data Science Diploka](https://s3-us-west-2.amazonaws.com/udacity-printer/production/certificates/59cca539-6553-4403-85cb-4d63be0101a2.svg)

