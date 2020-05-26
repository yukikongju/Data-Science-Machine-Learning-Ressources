### What is this repository about?
I document my learning process and the most interesting ressources I found to learn data science. 

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
1. Probability and distribution
2. Bayes Theorem. Used for naive Bayes classifier
3. Hypothesis testing

Calculus Checklist:
1. Integrals: calculate the area under a curve. Used for ROC and AUC
2. Derivatives: calculate the slope of a curve. Used for gradient descent
3. Partial Derivative and Chain Rule. Used for Backpropagation

Linear Algebra:
1. Vectors and Matrices operations: matrices multiplication, dot product, transpose, identity matrix. Used for data manipulation
2. Eighten values, unit vectors. Used for PCA, SVM, LDA, ...
3. Hyperplanes and distances: Used for K-Means Clustering, SVM, ...

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

Let's keep in mind that a Data Scientist has to be familiar with these 4 following subfields, altought they tend to specialize in one:
1. Data Mining
2. Data Visualization
3. Machine Learning
4. Statistical Analysis

### What is my learning style?

As of 2020-05-22, my learning method is somewhat arbitrary. I do some research on a subject, add it to one of the path, and learn it somewhere along the line. To be honest, I don't really have a way to determine which subject to learn next, but when I am choosing the ext subject to learn, I submerge myself completely. I don't really give myself a deadline, but I do keep track of my progress and fix goals periodically. I usually try to learn the theory first and then code.

### How did I build my curriculum?
As I see it, we can divide the data science learning process in four parts:

1. Computer Science tools with the command line
In this section, we focus on the tools that we use to write our code. We have to learn how to use an editor efficiently, how to do basic version control on git, and to use the command lines to search for files and data wrangling. I chose vim as my editor.

2. Data manipulation, visualization, statistical analysis with R
In this section, we focus on the statistical analysis, data visualization and data wrangling part of data science. Most of the code will be on R.

3. Modelling with Python
We create Machine Learning models and neural networks for data science. We build models for classification and regression. We explore computer vision and text generation. Later, we explore reinforcement learning and genetic algorithms, altought they are not required to learn.

4. Applications of Machine Learning
We go in-depth in the applications of machine learning: computer vision, language processing, speech recognition, and bioinformatics.

More: web applications and package creation. I have little experience in web development, but I did create a dashboard for data visualization in R with shiny.   

### Part 1: Get proficicient with your tools
1. MIT Missing Semester Course: https://missing.csail.mit.edu/2020/  https://missing.csail.mit.edu/2019/
2. Data Science at the Command Line by Jeroen Janssens: https://www.datascienceatthecommandline.com
3. Advanced Bash Usage Conference by James Pannacciulli at LinuxFest2017: https://www.youtube.com/watch?v=BJ0uHhBkzOQ&t=462s
4. Classic Shell Scripting by Arnold Robbins (focuses on data wrangling and file sear)

   Checklist:
   - General Commandline: pwd, cd, ls, mkdir, rm, rmdir, mv, cat, echo, head, tail, ...
   - Data Wrangling: sed, cut, join, sort, awk, uniq, fmt, wc, pr 
   - Regular Expressions:
   - Version Control: init, push, commit, merge, pull
   - Web Scrapping: curl, lynx
   - Basic Vim:
        1. Normal, Insert, Visual Mode
        2. Basic command: [dd,yy,p] [a,s,i,c] [gg,G] [w,b,W,B,e,gb] , more...
        3. Create new tab: [:tabnew, gt, gT] 
        4. Split screen: [:split, C-w + hjlk, ctrl+w ><]
        5. customize vim with .vimrc: NERDTree
        6. Recording Macros
    
### Part 2: Data Manipulation and Visualization with R
1. Basics of R
    1. Practice Book: R Programming for Data Science by Roger Peng: https://bookdown.org/rdpeng/rprogdatascience/
    2. Practice Book: R for Data Science by Hadley Wickham: https://r4ds.had.co.nz/
    
    Checklist:
    - import data
    - functions, if/else, for, while, apply
    - summary
    - column
     
2. Data Wrangling and cleaning with dplyr, tidyr, and stringr
    1. Practice: Data Wrangling from RStudio: https://www.youtube.com/watch?v=jOd65mR1zfw&list=PL9HYL-VRX0oQOWAFoKHFQAsWAI3ImbNPk
    2. [Strings manipulation with stringr] [todo]
    
    Checklist:
    - pipe command, %in%
    - dplyr functions: groupby, mutate, select, filter, summarise, aggregate, arrange, rename, count
    - tidyr: join, pivot_longer, pivot_wider, unite, separate
    - stringr: string manipulation
    
3. Data visualization with ggplot, plotly, ggAnimate, ggExtra, ggThemes, and more
    1. Data Visualization with R by Rob Kabacoff: https://rkabacoff.github.io/datavis/
    2. Data Visualization: an Introduction by Kieran Healy:https://socviz.co/index.html#preface
    3. Practice Book: R Graphics Cookbook by Winston Chang (focus on ggplot) or R Visualization Workshop: http://stulp.gmw.rug.nl/ggplotworkshop/ (focus on advanced plottind method)`
    4. Exploratory Data Analysis book by Roger Peng: https://bookdown.org/rdpeng/exdata/ (focus on clustering, dimensional reduction and plotting principles)
    
    Checklist:
    - Basic plots: geom_point(), geom_line(), geom_bar(), ...
    - More Plot Features with aes(): alpha, color, shape, size, ...
    - Grouping: facet_wrap, facet_grid
    - themes, axis, ...
    - Advanced plots: correlation plots, time-series, heat map, ...

### Part 3: Data Analysis with R

1. Introduction to Central tendencies, variation and correlation 
    1. Theory: Statistic Fundamentals Playlist by Statquest : https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9
    2. Theory: UC Business: http://uc-r.github.io/descriptive
    3. Practical Book: Applied Statistics in R: https://daviddalpiaz.github.io/appliedstats/applied_statistics.pdf

    Checklist:
    - Central Tendencies: Mean, median, mode, standard deviation,
    - Variation: What happens within a variables. Assumption (normalization), visualization (histogram and barplots), metrics (variance)
    - Correlation: relationship between variables. Assumptions (homogeneity), visualization (scatter plot, qqplot), metrics (R-squared)

2. Statistical Inferences and Hypothesis Testing 
   1. Theory: Statistics Playlist by Crash Course: https://www.youtube.com/playlist?list=PL8dPuuaLjXtNM_Y-bUAhblSAdWRnmBUcr

   Checklist:
   - confidence intervals, p-value, z-score, t-test, chi-square
   - statistical inference

### Part 4: Modelling with Python (and R)

0. Introduction to Python

[todo]

1. Machine Learning:
    1. Theory: Machine Learning Playlist by Statquest : https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
    2. Practice Book: Python Data Science Handbook by Jake VanderPlas (focus on data manipulation and data visualization)
    3. Practice Book: Machine Learning with Python by Chris Albon (focus on data preprocessing and all the ML algorithms)
    4. Practice Book: Introduction to Machine Learning with Python by Andreas Muller (focus on model visualization)
    5. Practice Book: Feature Engineering for Machine Learning by Alice Zheng (focus on featurization)
    6. Practice Book: Forcasting Principles and Practices with R: https://otexts.com/fpp2/ (focus on time-series and forecasting)
    
    Checklist:
    - Data Preprocessing: 
         a. Renaming Columns, Replacing Values, Delete Rows/Columns, Handling Missing values and outliers, Grouping columns/rows/dataframe
         b. Normalization and Standardization of Data:
               i. Numerical Data: rescaling, standardize, normalize, missing values, outliers
               ii. Categorical Data: Encode nominal and categorical features with OneHotEncoder, Imputer, ...
               iii. Text: cleaning, remove punctuations, tokenize, remove stop words, bag-of-words, weight word importance
               iv. Dates and Time:
               v. Images: Resize, Crop, Blurring, Enhancing, Edge/Corner Detection
    - Dimension Reduction for Feature Selection and Feature Extraction
    - Modelling:
         a. Linear Regression, Logistic Regression, K-Nearest Neighbor
         b. Ensemble: Trees, Random Forests
         c. Dimension Reduction: PCA, LDA, K-Means Custering, DBSCAN, Hierarchical Clustering
         d. Suport Vector Machine, MDS and PCoAs, 
         e. Naive Bayes
    - Model Evaluation and Selection with with Exhaustive and Randomized Search 
    - Training Faster: Parallelization
    - Visualizing Models

2. Neural Network:
    1. Theory: MIT Deep Learning Playlist by Alexander Amini: https://www.youtube.com/watch?v=njKP3FqW3Sk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI
    2. Practice Book: Deep Learning with Python by Francois Chollet (focus on basics Neural Networks)
    
    Checklist:
    - Neural Network Models:
         a. Multi-layers Perceptrons
         b. Convolutional Neural Network
         c. Recurrent Neural Network, LSTM
         d. Autoencoders, GAN, GRU
    - Functions:
         a. Loss: sparse_categorical_crossentropy,categorical_crossentropy, binary_crossentropy, mse, mae
         b. Activation: sigmoid, softmax, relu, adam, ...
         c. optimizer: rmsprop, sgd, ...
    - Hyperparameters Tuning with GridSearch
    - Using Tensorboard and callbacks
    - concatenate layers for multi inputs/outputs models

3. Reinforcement Learning:
    1. Theory: Reinforcement Learning Course given by David Silver on Deep Mind. In-depth explanation on agent, environment, decision processes and policies. Link: https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ

4. Genetic Algorithms:
    1. Theory: Genetic algorithm by The Coding Train - 9.2. Great Overview of fitness function, parents selection, crossover, and mutation. Link: https://www.youtube.com/watch?v=RxTfc4JLYKs

### Part 5: Data Mining with R and the command line

1. Data Scrapping with rvest 
    1. Code: Using Chrome Extension: https://www.youtube.com/watch?v=4IYfYx4yoAI&t=9s
    2. Code: Scrapping Table: https://www.youtube.com/watch?v=0mvlZhYk44E&t=912s

    Checklist:
    - get data table from website
    - get text data from website

### Part 6: Applications of Machine Learning 

Tutorial for Bioinformatics:
1. Book: Deep Learning for Life Science by Peter Eastman

      Checklist:
      - Central Dogma

      Ressources:
      - Scitable by Nature Education: https://www.nature.com/scitable/topics/
      
      Modules:
      - DeepChem
      - Biopython
