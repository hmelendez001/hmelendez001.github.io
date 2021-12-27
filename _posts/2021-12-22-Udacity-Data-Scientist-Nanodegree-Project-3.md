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
### What is this Project?
For this project I analyze the interactions users have with articles on the IBM Watson Studio platform and make recommendations to them about new articles I think they will like. This will be accomplished by creating a Recommendation Engine that employs a number of the techniques taught in this program. We will use existing data that gives us usage history to create a Matrix Factorization technique called Funk SVD or Simon Funk's Singular Value Decomposition method which allows for incomplete or missing data points. Interesting sidebar, Funk SVD was part of the competition in coming up with the Netflix Recommendation Engine example above, see <a href="https://sifter.org/~simon/journal/20061211.html" target="_blank">Netflix Update: Try This at Home</a> for more on this. Also predicting the effectiveness of our model, rank-based (top ratings by all, most liked by all, etc.), knowledge-based (adding filters where the user can help us pinpoint their preferences like publication year, subject matter, genre, etc.), and collaborative filtering (match to what other similar users liked).

<img src='https://video.udacity-data.com/topher/2018/September/5ba02d6d_screen-shot-2018-09-17-at-3.40.30-pm/screen-shot-2018-09-17-at-3.40.30-pm.png'/>
Source: Udacity.com IBM Watson Studio Community Board Screenshot

---
### Results
This project was all based on a Jupyter Notebook and the resulting analysis. As with the previous project, I have learned I needed to get my Jupyter Notebook working outside of the Udacity virtual environment to really test it. So I made sure to download the Jupyter Notebook runtime to my local laptop and ran the final results from there, otherwise, there were some runtime errors I would have missed. For example, I used a wordcloud Python component that does not work out of the box for the version of Python I am using. It gives you a C++ compile time error on installation. So instead you have to use the "Wheel" compiled version and this required me to include this file (wordcloud-1.8.1-cp39-cp39-win_amd64.whl) with my source code as well as installing it right after I installed matplotlib at the top of my notebook:

```
     # From: https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/
     # Install a pip package in the current Jupyter kernel
     import sys
     !{sys.executable} -m pip install matplotlib
     !py -3.9 -m pip install wordcloud-1.8.1-cp39-cp39-win_amd64.whl
```

---
#### Results: Part One
In the first part of our project we do the analysis of the IBM user article interaction data. We clean up some duplicates and answer some initial questions like how many unique articles and users we have in the data or the number of interactions recorded. This is all just preparation to get to the actual recommendation engine techniques that follow.

<img src='https://user-images.githubusercontent.com/26253570/147491507-02b85e8b-85d1-45c5-83e4-504cfd385503.png'/>

---
#### Results: Part Two
In the second part we implement Rank-Based recommendations to answer what are the top 10 articles or the top 20 articles, etc. Rank-Based recomendations are just what the name implies where we simply find the most popular articles. In finance or hedge fund world we would typically call these "top 10 winners" and "top 10 losers" because you wanted to know both the ceiling and the floor, but same idea.

<img src='https://user-images.githubusercontent.com/26253570/147491516-39cf6b11-1fec-46c2-a945-6d32f5d391ce.png'/>

---
#### Results: Part Three
Part 3 gets more interesting. We develop User-User Based Collaborative Filtering to find similar users based on articles they have interacted with. This is more like the Netflix recommendation engine: other users that have interacted with this article have also interacted with these other articles that you have not seen yet. Then we improve the consistency of our engine. Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user, we choose the users that have the most total article interactions before choosing those with fewer article interactions.

<img src='https://user-images.githubusercontent.com/26253570/147491530-74169ba5-b50b-4d44-83b5-79fba51c75a5.png'/>

---
#### Results: Part Four
And then in part 4 we implement Content Based Recommendations. Based on the title of our articles we select the top terms in the title and create content likeness based on the appearance of these terms. Then we use this in a matrix of users and content to further match users. In order to get the top terms we just used a word cloud but concluded that we could have also gotten top terms programmatically to better determine the number of terms to best use. This would have been better in the end than arbitrarily choosing the top 10, 15, or 20 terms, but for our purposes we proved our point on being able to create Content Based Recommendations. Also, we had one false start here. We started with the dataset that included article document full name and description, but quickly realized this data had over half the content we needed missing, so instead we went with just the article titles, good enough for our analysis.

<img src='https://user-images.githubusercontent.com/26253570/147491528-424a81d4-9570-4f73-a373-847816b41bc3.png'/>

---
Word clouds were simple to generate using the Python wordcloud module. Where we ran into issues was when we tried to deploy this outside of the Udacity classroom VM. Turns out that based on the version of Python you are running, the underlying module relies on another module called Wheel. That module will not compile the underlying C++ code if you are running a more recent version of Python, so you have to include the compiled version. But once we resolved this the results were great and easy to generate:

<img src='https://user-images.githubusercontent.com/26253570/147492770-1703023c-ecd6-4c7c-b0f8-154e7b732031.png'/>

---
#### Results: Part Five
Finally, the last part, part 5 involved Matrix Factorization, or specifically Singular Value Decomposition or SVD. Now we're cooking on a cool recommendation engine the likes of Netflix and Amazon and all the big e-retailers. This involved matrix math with eigen values and eigen vectors. We also addressed the Cold Start Problem. This is the common recommendation engine problem: what to do with a new user who has no history or new content (a new movie, a new product, or in our case a new IBM article) that has no history. One way is to eliminate those for the purpose of using pure SVD, but this is where Funk SVD shines because it addresses exactly: SVD with missing values.

<img src='https://user-images.githubusercontent.com/26253570/147491529-ee216a14-5e10-4277-8e15-60fa77b50759.png'/>

---
### Conclusion
Although I enjoyed the content of this lesson, discovering how recommendations work and building one of my own, the project itself was a little too academic for me. The other projects so far have been the opposite. The lesson content was tedious at times but the resulting project produced something with more real world application. For this project it was too close to the lesson. Although it was probably the easier of the projects so far, I found I did not enjoy it as much and I was more excited about moving onto my final capstone project where I can produce a final web application with some fun content to share and show off what I've learned in taking this nanodegree course.
