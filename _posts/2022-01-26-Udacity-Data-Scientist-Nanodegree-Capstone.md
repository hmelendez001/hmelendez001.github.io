## Udacity Data Scientist Nanodegree: Project 4 Final Capstone Project

As part of the udacity.com Data Scientist nanodegree, this is project four and the final project of the program, otherwise known as our Capstone Project. This project involves building a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. The inputs will contain multiple metrics, such as opening price (Open), highest price the stock traded at (High), how many stocks were traded (Volume) and closing price adjusted for stock splits and dividends (Adjusted Close); we only need to predict the Adjusted Close price.

Investment firms, hedge funds, and even individuals or day traders have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process. According to an article by Mark Kolakowski ["How Robots Rule the Stock Market (SPX, DJIA)"](https://www.investopedia.com/news/how-robots-rule-stock-market-spx-djia):

> only 10% of trading volume now comes from human discretionary investors, per data from JPMorgan Chase & Co. (JPM) cited by Bloomberg

![Crypto Trading Robot Image](https://insidebitcoins.com/wp-content/uploads/2020/05/automated_forex_robot-1200x900-1-768x576.jpg)
##### Source: "Best Forex Robots 2021" by Kane Pepi at https://insidebitcoins.com/forex-robot

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
##### Source: "SPACs break 2020 record in just 3 months, but the red-hot industry faces challenges ahea" by Yun Li at https://www.cnbc.com/2021/03/19/spacs-break-2020-record-in-just-3-months.html
###### Traders work during the opening bell at the New York Stock Exchange (NYSE) on March 5, 2020 at Wall Street in New York City. (Photo by Johannes EISELE / AFP) (Photo by JOHANNES EISELE/AFP via Getty Images)
###### Photo by Johannes Eisele | AFP | Getty Images

---
### Data Mining
For my primary market data source I considered a few suggestions like [Yahoo! Finance API](https://www.yahoofinanceapi.com), [Bloomberg API](https://www.bloomberg.com/professional/support/api-library/), and [NASDAQ Financial API](https://data.nasdaq.com/tools/api) as per our [Udacity Project Description](https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub). My Wall Street background immediately drew me to Bloomberg as the obvious choice, but there are pricing and licensing limitations that come with Bloomberg. So I decided on Yahoo! Finance mainly because there was no need to sign up and I found a lot of online references for Python rather easily. It offers an API that is easy to use and well documented. So that is what ultimately drove my decision. I downloaded the data with the following code and saved the csv file for the analysis:

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
APIs and specially Financial Markets APIs like Yahoo! Finance have limits on how much data you can pull and how frequently. I will minimize the number of times I run code like the one above to avoid hit these limits, and heaven forbid, get rate limited or blacklisted. I will show how we can cache requests as much as possible so that I am not hitting the API as frequently. Since this is an academic exercise I am not concerned with real-time accuracy just data to run the models against, so data is cached on a per day basis: meaning, I call the API once per symbol and date range, and store the results in a database so as to minimize the amount of times we call the API.

**My disclaimer: I am not a certified financial planner nor a financial advisor nor a certified financial analyst nor an economist nor a CPA. These predictions and recommendations are only for academic purposes and not an official recommendation from Udacity nor from my employer RSM in any way.**

![Yahoo! Finance API](https://algotrading101.com/learn/wp-content/uploads/2020/05/yahoo_homepage-2-1024x545.png)
##### Source: "Yahoo Finance API – A Complete Guide" by Greg Bland at https://algotrading101.com/learn/yahoo-finance-api-guide/

---
#### Other Financial Indicators
As part of the data wrangling, I also wanted to bring in other factors I feel have an effect on stock prices in general. Some of this data like the Department of Labor Weekly Unemployment Numbers will require some wrangling to extrapolate a daily number since it is published weekly and some will be easy to both acquire and merge with our stock price and volume data because it is market data that trades with a stock ticker. So I also pulled that down and saved to the data folder like with the stock price data.

1. Weekly Unemployment Numbers from https://oui.doleta.gov/unemploy/claims.asp: unemployment numbers typically drive market numbers. I can remember every Thursday when working on Wall Street what a big deal it was when these numbers came out from the US Department of Labor<br>
2. CPI or Consumer Price Index: how well is the consumer market spending is a typical bell weather of consumer confidence<br>
3. DJIA or Dow Jones Industrial Average Index: an overall market indicator<br>
4. Google Global Mobility Report from https://www.google.com/covid19/mobility: how much are people getting around in spite of the COVID-19 global pandemic, filtered down to just US numbers<br>
5. SP&500 Index: ^GSPC ticker: additional market indicator<br>
6. NASDAQ Composite Index: ^IXIC ticker: Over the Counter market indicator<br>

![DJIA: Dow Jones Industrial Average](https://cdn.corporatefinanceinstitute.com/assets/Dow-Jones-Industrial-Average-1.jpeg)
##### Source: "What is the Dow Jones Industrial Average (DJIA)?" at https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/dow-jones-industrial-average-djia/

---
#### Data Preprocessing
The Weekly Unemployment Numbers came as an Excel file, so I opened it in Excel and saved it as a CSV for our convenience: data/DOL_Weekly_Unemployment_US_2021.csv.

For the Google Global Mobility Report, I read in the global report into a Pandas dataframe because this file was too big to simply open in Excel like I did for the Weekly Unemployment Numbers.

```
import pandas as pd
import numpy as np

df = pd.read_csv('data/Global_Mobility_Report.csv')

# Show df to get an idea of the data
df.head()
```

The I noticed it was as expected but more than US national numbers: US state numbers. And also more than just 2021. So I took just the data I needed from this data frame and wrote it out for later use:

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
For the others I kept the format as is in the data folder. So for the analysis that leaves me the following 7 files to merge:

1. Weekly Unemployment Numbers: data/DOL_Weekly_Unemployment_US_2021.csv<br>
2. CPI or Consumer Price Index: data/cpi_index_2021.csv<br>
3. DJIA or Dow Jones Industrial Average Index: data/djia_index_2021<br>
4. Google Global Mobility Report: data/Global_Mobility_Report_US_2021.csv<br>
5. SP&500 Index: data/sp500_index_2021.csv<br>
6. NASDAQ Composite Index: data/nasdaq_composite_index_2021.csv<br>
7. Bitcoin USD Ticker Data: data/bitcoin_stock_2021.csv<br>

### Initial Findings
A correlation analysis using the Weekly Unemployment numbers revealed no correlation, at least for our USDB Bitcoin ticker, so we also checked against other equities (Tesla Motors and Texas Instruments, but same results. Perhaps the Weekly Unemployment has correlation to the Fixed Income market (bonds, loans, structured debt, etc.), but not the Equity names I analyzed. In fact, none of the economic indicators we anticipated to correlate to our stock prices actually correlated in a significant way.

![Market Indicator Correlation Analysis](https://user-images.githubusercontent.com/26253570/148664284-ca49412e-9437-49a9-b1a7-4ddaa155745e.png)

---
Okay, some positive correlation, but also some negative effects from the following: 

`a.` parks_percent_change_from_baseline: people going to the park or not apparently does not effect Bitcoin value<br>
`b.` residential_percent_change_from_baseline: interesting there is negative correlation since more people are working from home during the pandemic<br>
`c.` S&P Volume: Volume of blue chip trading has no correlation<br>
`d.` NASDAQ Volume: Neither does volume of over the counter trading at the NASDAQ<br>

So let's update our status table...

| No | Indicator | Data Source | Description | Status |
|----|-----------|-------------|-------------|--------|
| 1 | Weekly Unemployment | data/DOL_Weekly_Unemployment_US_2021.csv | I can remember every Thursday when working on Wall Street what a big deal it was when these numbers came out from the US Department of Labor | DONE, no correlation |
| 2 | CPI or Consumer Price Index | data/cpi_index_2021.csv | Indicates strength of consumer market spending, a typical bell weather of consumer confidence | DONE, correlation found |
| 3 | DJIA or Dow Jones Industrial Average Index | data/djia_index_2021.csv | Overall blue chip market indicator | DONE, correlation found |
| 4 | Google Global Mobility Report | data/Global_Mobility_Report_US_2021.csv | How much are people getting around in spite of the COVID-19 global pandemic, filtered down to just US numbers | DONE, some correlation found with the exception of **points a and b** above |
| 5 | SP&500 Index | data/sp500_index_2021.csv | ^GSPC ticker: additional market indicator | DONE, some correlation found with the exception of **point c** above |
| 6 | NASDAQ Composite Index | data/nasdaq_composite_index_2021.csv | ^IXIC ticker: Over the Counter market indicator | DONE, some correlation found with the exception of **point d** above |

...and even after I re-ran my model with this newfound knowledge, I concluded that with less work, i.e. no economic indicator data, we can do a better job of predicting future stock value. Our linegar regression model was a good start using just the historical data provided by Yahoo! Finance on the stock symbol.

### Implementation - The Stock Predictor
First, I created some helper functions and classes that would be needed for validating and pre-processing the inputs. I also included some unit testing with these just as a sanity check. This helped me make changes more confidently as well as refactor code as needed while being able to run some quick regression tests to make sure everything was working as expected:

`1.` Function `is_valid_range(from_date, to_date, can_be_in_the_future = False)`: Is the given date range a set of valid dates, by default they have to be in the past or historical dates, but for making future prediction the date range can be in the future<br>
`2.` Function `def get_from_cache(symbol, from_date, to_date)`: Check for symbol and validate ranges to see if we have cached them already. Even invalid symbol calls to the API were cached so that we would not call invalid symbols for the purpose of regression testing more than once per run<br>
`3.` Function `get_stock_history(symbol, from_date, to_date)`: This was the function that actually checked the cache and then called the Yahoo! Finance API to get historical data as needed<br>
`4.` Class `TrainedModel`: At first this was a class I created to encapsulate and cache both the model as well as associated parameters like training and test dates, etc., but then I realized I could use this abstract base class to switch in and out the type of underlying machine learning model in the future<br> 
