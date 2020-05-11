### What is data science?

Data science combines the field of statistics and programmation to describe data, predict values and prescribe recommendations based on the predictions made in a specific domain, such as business analytics, sport analytics and bioinformatics. It helps us understand how the decisions we take influence our world, quantify if the changes implemented have been helpful, and suggest how should we correct our comportement to reach our goals. 

[to continue]

### Understanding a typical data science lifecycle 

1. Define the problem: we want to determine what question we do we want to answer, what observations do we want to observe and what variables do we need to keep track of.

2. Gather the data: we want to get the data required to perform our analysis. Depending on our time and ressources, we may want to create our own dataset by conducting our survey or creating a customer data pipeline, or use an already existing dataset. We can also scrape the data from the web.

3. Data Wrangling: we clean the data so that the strings are readable, manipulate the dataset so that the data is compact, handle missing values and manage outliers.

4. Data exploration: we perform a descriptive analysis to understand the relationship within and between the variables statistically and visually. 

5. Create the model: based on our initial goals and what we have discovered in the exploratory phase, we create a model to predict, classify or cluster our data. To create our model, we need to preprocess our training data, fit our model with our training data, evaluate our model with our testing data, and optimize it using regularization techniques. We create a set of models and choose the one that best fits our needs.

6. Deploy the model: [to do]

7. Refine the model: [to do]

### Some questions I had when I first started learning Data Science?

##### What programming languages should I learn?

There are several programming languages, but the best ones for data science are Python and R in my opinion. Python has more emphasis on machine learning and is better for modelling, whereas R is more suited for statistical analysis, data visualization and data manipulation, so I think that learning a combination of both would be the way to go. Regarding data cleaning, both languages offer basics strings reformatting functions, but I found that using the UNIX command line is the best way to go, as it requires less lines of code and computation power to perform the same task.

Additionally, if you do plan to become a data scientist, you will need to know how to manipulate a large database, so you'll need to learn the basics of SQL.

##### What maths concepts do I need to know?

One should have strong statistical, calculus and linear algebra knowlegde to master data science. Statistical analysis is mostly used in the descriptive analysis and data exploration process, where as calculus and linear algebra are mostly used for modelling. 

Statistics Checklist:
[to do]

Calculus Checklist:
[to do]

Linear Algebra:
[to do]

##### How can I apply data science in real life?

##### How do I get started on my data science journey?

From my experience, there are 3 principal specialization in data science:

- Data exploration and visualization: Mostly focus on statistical analysis and creating plot. Emphasis on R and statistical concepts

- Modelling: Create models for prediction, classification and clustering using the basic machine learning algorithms and neural networks. Could also use reinforcement learning and genetic algorithms althought not too popular. Should emphasize on Python

- Web applications and package creations: Build a product for other people to use. Requires more computer science background and programming fundamentals.

The order in which you learn data science greatly depends on which of these 3 fields interest you the most. If data exploration and visualization interest you the most, you should spend most of your time mastering statistics and visualization techniques on R. If modelling algorithms is what interest you most, you should learn Python and the main machine learning package: numpy, pandas, matplotlib, keras and tensorflow. If creating web applications and packages is your cup of tea, then you should invest your time in deployment. I have less experience in that field, so I can't comment much on that.

Personnally, I would suggest starting with statistical understanding and data visualization as it doesn't require as much mathematical knowledge as modelling, nor specific field knowledge to understand the business needs requires to construct a web applications.

### What this repository is about?

I document my learning process and the most interesting ressources I found to learn data science. In this repository, I will mostly focus on the statistical, analytics and visualization part of data science. Most of the code will be on R.

I have also documented my machine learning process in my "Machine-Learning-Ressources" repository. However, I have yet to learn the web application process. I only did a quick web application in R using the shiny package.

### How I assembled my curriculum?

##### 0. Basics of R

[to do]

##### 1. Data Wrangling with dplyr and tidyr

dplyr functions: groupby, mutate, select, filter, summarise, aggregate, arrange, rename, count
tidyr: join, pivot_longer, pivot_wider, unite, separate

[todo]

##### 2. Data visualization with ggplot

ggplot: 
more package: plotly, ggAnimate, ggExtra, ggThemes

[to do]

##### 3. Introduction to Central tendencies, variation and correlation

Part 1: Central Tendencies

Mean, median, mode, standard deviation, 
[to do]

Part 2: Variation: What happens within a variables

1. Assumptions: Normalisation
2. Visualization: Histogram and barplots
3. Metrics: Variance

[to do]

Part 3: Correlation: relationship between variables

1. Assumptions:
2. Visualization: scatter plot, qqplot
3. Metrics: R-squared, 

##### 4. Statistical Inferences and Hypothesis Testing

confidence intervals, p-value, z-score, t-test
[todo]
