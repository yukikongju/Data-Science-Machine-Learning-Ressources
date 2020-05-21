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

Ressources: 
1. Decisive Data: https://www.youtube.com/watch?v=24G_pfcl3qE&list=PLwM2SFDcolcK9WIpn50JkELC0grYP5Pyu

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

Ressources:
1. The Elements of Statistical Learning: https://web.stanford.edu/~hastie/Papers/ESLII.pdf

##### How can I apply data science in real life?

[to do]

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

##### 0. Get proficient with your tools

- Shell Scripting and Command Line
- Data Wrangling in Bash
- Version Control (git)
- Get familiar with editors: vim

Learning Path:
1. General Command: pwd, cd, mkdir, rm, rmdir, mv, cat, echo, head, tail
2. Basic Vim:
  2.1. Normal, Insert, Visual Mode
  2.2. Basic command: [dd,yy,p] [a,s,i,c] [gg,G] [w,b,W,B,e,gb] , more...
  2.3. Create new tab: [:tabnew, gt, gT] 
  2.4. Split screen: [:split, C-w + hjlk, ctrl+w ><]
  2.5. customize vim with .vimrc: NERDTree
  2.6. Recording Macros
3. Advanced commands: sed, cut, join, sort, awk, uniq, fmt, wc, pr 
  3.1. Book: Classic Shell Scripting by Arnold Robbins 
4. Regular Expressions:
5. In depth sed & awk:
6. Version Control

General Ressources:
1. MIT Missing Semester Course: https://missing.csail.mit.edu/2020/  https://missing.csail.mit.edu/2019/
2. Data Science at the Command Line by Jeroen Janssens: https://www.datascienceatthecommandline.com
3. Advanced Bash Usage Conference by James Pannacciulli at LinuxFest2017: https://www.youtube.com/watch?v=BJ0uHhBkzOQ&t=462s

##### 1. Basics of R

Ressources:
1. R Programming for Data Science by Roger Peng: https://bookdown.org/rdpeng/rprogdatascience/
2. R for Data Science by Hadley Wickham: https://r4ds.had.co.nz/
[to do]

##### 2. Data Wrangling with dplyr and tidyr

- dplyr functions: groupby, mutate, select, filter, summarise, aggregate, arrange, rename, count
- tidyr: join, pivot_longer, pivot_wider, unite, separate
- data wrangling in bash

Ressources:
1. Data Wrangling from RStudio: https://www.youtube.com/watch?v=jOd65mR1zfw&list=PL9HYL-VRX0oQOWAFoKHFQAsWAI3ImbNPk

[todo]       

##### 3. Data visualization with ggplot

- ggplot: 
- more package: plotly, ggAnimate, ggExtra, ggThemes

Ressources: 
1. Data Visualization with R by Rob Kabacoff: https://rkabacoff.github.io/datavis/
2. Data Visualization: an Introduction by Kieran Healy:https://socviz.co/index.html#preface

[to do]

##### 4. Introduction to Central tendencies, variation and correlation

Part 1: Central Tendencies

- Mean, median, mode, standard deviation, 

Ressources: 
1. Statistic Fundamentals Playlist by Statquest : https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9
2. UC Business: http://uc-r.github.io/descriptive

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

##### 5. Statistical Inferences and Hypothesis Testing

- confidence intervals, p-value, z-score, t-test

[todo]

Ressources: 
1. Crash Course: https://www.youtube.com/playlist?list=PL8dPuuaLjXtNM_Y-bUAhblSAdWRnmBUcr

##### 6. Data Scrapping with rvest 

- get data table from website
- get text data from website
- use lynx on bash

Ressources:
1. Using Chrome Extension: https://www.youtube.com/watch?v=4IYfYx4yoAI&t=9s
2. Scrapping Table: https://www.youtube.com/watch?v=0mvlZhYk44E&t=912s

[todo]

##### 7. Modelling

- Machine Learning
- Neural Network
- Reinforcement Learning
- Genetic algorithm

Ressources:
1. Theory: Machine Learning Playlist by Statquest : https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
2. Theory: MIT Deep Learning Playlist by Alexander Amini: https://www.youtube.com/watch?v=njKP3FqW3Sk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
3. Book: Machine Learning with Python by Chris Albon (focus on data preprocessing and all the ML algorithms)
4. Book: Deep Learning with Python by Francois Chollet (focus on basics Neural Networks)

4. Book: Deep Learning with Python by Francois Chollet (focus on neural network and their applications):
3. Book: Forcasting Principles and Practices: https://otexts.com/fpp2/
