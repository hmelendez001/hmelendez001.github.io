## Udacity Data Scientist Nanodegree: Project 3

As part of the udacity.com Data Scientist nanodegree, this is project three of the program.

This project involves Recommendation Engines. These are the algorithms used to make suggestions or recommend things to you based on past purchases or usage, or based on what other users are buying or downloading. For example, think of early Point of Sales systems that would tell cashiers to make suggestive sales like “if they are purchasing nails suggest a new hammer as well.” Nowadays, apps like Netflix have Recommendation Engines in place which suggest movies that you might also enjoy based on what you have been streaming or enjoying in the past or based on what is most popular with other users.

<img src='https://media-cldnry.s-nbcnews.com/image/upload/t_fit-2000w,f_auto,q_auto:best/streams/2013/November/131113/2D9672564-attachment.jpg'>
Source: https://www.nbcnews.com/technolog/netflix-makeover-brings-improved-search-recommendations-2d11582698 Netflix makeover brings improved search and recommendations by Devin Coldewey

---
Other popular retail sites using these types of algorithms include [Amazon.com](https://www.amazon.com/) or [Target.com](https://www.target.com/). And apps like Starbucks.com which have also joined the race to earn your e-commerce dollars using both recommendation engines and offers, specials, and loyalty rewards.

<img src='https://images.squarespace-cdn.com/content/v1/596ce3c8cd0f68df1fd7598e/1623345003169-N7A7P2KTKSZT074HMXC6/Screen+Shot+2021-06-10+at+1.07.45+PM.png?format=1500w'>
Source: https://www.indigo9digital.com/blog/starbucksmobileapps How Starbucks is Using Mobile Apps to Significantly Increase Sales by Tricia McKinnon

---
For this project I analyze the interactions users have with articles on the IBM Watson Studio platform and make recommendations to them about new articles I think they will like. This will be accomplished by creating a Recommendation Engine that employs a number of the techniques taught in this program. We will use existing data that gives us usage history to create a Matrix Factorization technique called Funk SVD or Simon Funk's Singular Value Decomposition method which allows for incomplete or missing data points. Interesting sidebar, Funk SVD was part of the competition in coming up with the Netflix Recommendation Engine example above, see <a href="https://sifter.org/~simon/journal/20061211.html" target="_blank">Netflix Update: Try This at Home</a> for more on this. Also predicting the effectiveness of our model, rank-based (top ratings by all, most liked by all, etc.), knowledge-based (adding filters where the user can help us pinpoint their preferences like publication year, subject matter, genre, etc.), and collaborative filtering (match to what other similar users liked).

<img src='https://video.udacity-data.com/topher/2018/September/5ba02d6d_screen-shot-2018-09-17-at-3.40.30-pm/screen-shot-2018-09-17-at-3.40.30-pm.png'>
Source: Udacity.com IBM Watson Studio Community Board Screenshot

---