
# Introduction
=====================================================================================================

   Between the night of April 14, 1912 and the morning of April 15, 1912, the deadliest peacetime maritime disaster in history occured. At the time, the RMS Titanic was considered the largest and most lavish passenger liner in existance. You'd be hard-pressed to find anyone born in the United States that is unfamiliar with the events of the Titanic, making it an icon of history. Because it was such an unprecented tragedy, it has been portrayed in many instances of popular culture, including a film. Because of its appearences in popular culture, this presentation will not provide the history behind the sinking of the RMS Titanic any further than what already has been stated. What this presentation **will** provide, however, is some insight into who survived the Titanic and who did not.
   
#### Purpose/Questions
   
   The purpose of this presentation is to provide any insight into what variables made someone more or less likely to survive the sinking of the RMS Titanic. It provides some descriptive statistics of select variables, a barplot with 95% confidence intervals, a violin plot, and a histogram. This presentation also provides a few statistical models to predict the probability of survival for a certain passenger. A link to the original dataset is provided in the 'variables description' section of this presentation. **I must make this very clear, any causal claims about the data is far beyond the scope of this presention**. I have attempted to make my statistical models as robust as possible, but this presentation is clearly for exploration and fun. I do not pretend to be a PHD in statistics, or any sort of expert. That said, I am well pleased with the results.
   
#### Methods

   The statistical methods applied to this dataset will be described further in the 'Exploration' section of this presention, but a brief overview will be provided here. First, we will apply the ordinary least squares regression model with robust standard errors (normal standard errors assume errors are homoskedastic) and multiple regressors. The dependent variable is a binary variable for wether or not a passenger survived, while the main independent variable of interest is another dummy variable for wether or not a passenger is female (1 for female, 0 for male). This is not a conventional way to use OLS, because usually the dependent variable is continuous. For this reason, the probit method is later applied, with survivorship again as the dependent variable and two regressors(ticket fare and gender) to see if an increase in ticket fair significantly increases a passengers chance to survive.  


```python
import pandas as pd
import numpy as np

titanic_data = pd.read_csv('/Users/vincentshields/Desktop/Udacity/titanic-data.csv') #Read the csv into a pandas dataframe
```

# Data Wrangling
=======================================================================================================

### Cleaning the data:

Since this data set seems to be used by teachers and universities all around the globe, the data was already quite clean. However, there were a few changes that had to be made in order to make the data more presentable for the purposes of this presentation. First, the data in the column "Age" had to be changed from floats to integers. It doesn't make sense to leave age as a float, because age is almost always measured in years. If Age was left as a float it could cause some data visualizations to appear misleading. The "Sex" column was changed to binary data rather than "male"/"female" string types. Next, this new column was renamed "Female" and the old "Sex" column was dropped in order to avoid me accidentaly including both of them in a regression. A dummy variable for wether or not a passenger had children or parents on board was also created. A function to calculate the birth year for each passenger, based on their age and the year the titanic sank(1912) is used to create the variable "birth year". This was mostly just for curiosity. Most importantly, variables representing the number of siblings and spouses and parents and children for each passenger were combined into one variable, total family.  


```python
def get_birth_year(age):
    """Returns the birth year for a given passenger"""
    return 1912 - age


titanic_data["Age"] = titanic_data["Age"].fillna(0.0).astype(int) #Change the age collumn from floats to integers and fill missing values
titanic_data["Sex"] = pd.get_dummies(titanic_data.Sex)["female"] # Change the Sex column to take on numeric form
titanic_data["Female"] = titanic_data["Sex"] # Rename the new column
del titanic_data['Sex'] # drop the "sex" column to avoid multicolinearity 

titanic_data['has_cpob'] = titanic_data['Parch'] > 0 # creating a new variable has children on board
titanic_data['bith_year'] = get_birth_year(titanic_data['Age']) # This was mostly just for curiosities' sake
titanic_data['total_fam'] = titanic_data['SibSp'] + titanic_data["Parch"] 
#These two variables are combined to create a total_fam variable.


```

### Addressing missing values:


```python
# Checking for missing values

print "there are",titanic_data["Fare"].isnull().sum(),"missing values in Fare" 
print "there are",titanic_data["Female"].isnull().sum(),"missing values in Female"
print "there are",titanic_data["Age"].isnull().sum(),"missing values in Age"
print "there are",titanic_data["total_fam"].isnull().sum(),"missing values in total_fam"

```

    there are 0 missing values in Fare
    there are 0 missing values in Female
    there are 0 missing values in Age
    there are 0 missing values in total_fam


The variable "Age" originally had some missing values, which was handled by using the .fillna(0.0) method seen in the second cell block. This seems to be more reasonable than dropping that row completely, because it is reasonable to assume that missing values exist in the age column for a given passenger because that passenger is a toddler or very young. All other variables used in the statistical models seen later have zero missing values.


```python
# Check SibSp and Parch for multicolinearity

def pearsons_r(var1 , var2):
    """Calculates pearsons r between two variables"""
    var1_z = (var1.mean() - var1 / var1.std(ddof=0))
    var2_z = (var2.mean() - var2 / var2.std(ddof=0))
    return (var1_z * var2_z).mean()

print "The correlation between Sibsp and Parch is: ",pearsons_r(titanic_data["SibSp"], titanic_data["Parch"])
```

    The correlation between Sibsp and Parch is:  0.410375261998


The correlation between Sibsp and Parch is closer to zero than it is to one, so including both these variables in a regression probably would not cause any issues with multicolinearity. However, it still seems to make sense to combine these two variables. 

### Variable Descriptions:

**Survived:**      0 if the passenger died 1 if the passenger survived





**Pclass:**        The ticket class the passenger belonged to ( 1 to 3)




**Name:**          The Passengers full name, if the passenger is female the maiden name is usually included.




**Age:**           Age of the passenger in years




**SibSp:**         Number of siblings or spouses on board


**Parch:**         Number of parents or children on board


**Ticket:**	       Ticket number	


**Fare:**	       Passenger fare	


**Cabin:**	       Cabin number	


**Embarked:**      Port of Embarkation   C = Cherbourg, Q = Queenstown, S = Southampton


**Female:**        0 if the passenger is male and 1 if the passenger is female



**has_cpob:**      True if the passenger has any family on board



**birth_year:**    The year the passenger was born 




**total_fam:**     Total number of family members on board

For a detailed description of the original dataset, please visit [this](https://www.kaggle.com/c/titanic/data) webpage. 



# Exploration
=================================================================


```python
titanic_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Female</th>
      <th>has_cpob</th>
      <th>bith_year</th>
      <th>total_fam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>False</td>
      <td>1890</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>False</td>
      <td>1874</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>False</td>
      <td>1886</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
      <td>False</td>
      <td>1877</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>False</td>
      <td>1877</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Here is a brief look at the data. You can see some of the variables from the original dataset as well as a few that were created with Pandas. Even a small glance into the pandas dataframe piques my personal interests. The dataset provides the full name of the passenger, which makes it even more relatable. I found an interesting interview from 1957 with some survivors of the Titanic. Perhaps [it](https://www.youtube.com/watch?v=FVLiZo6Pkak) will provide more excitement about exploring the data.


```python

titanic_data['Fare'].describe() 
```




    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: Fare, dtype: float64



The median ticket fare is about 14.5 dollars and the third quartile is exactly 31 dollars. This is important for the probit model seen later. The 0 dollar minimum may be due to workers or young children travelling with their family. 


```python
titanic_data.groupby('Survived')["Age","Fare","total_fam"].mean()  

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>total_fam</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.637523</td>
      <td>22.117887</td>
      <td>0.883424</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24.017544</td>
      <td>48.395408</td>
      <td>0.938596</td>
    </tr>
  </tbody>
</table>
</div>



The average age of a non-survivor versus a survivor appear very similar. The same is true for the total_fam variable. The only stark difference is between the average fare for a non-survivor(22) versus a survivor(48). Is there a strong positive relationship here? Any conclusions would be so far premature.


```python
%matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
plt.figure(figsize=(10,8))
sns.barplot(data=titanic_data, x="Female", hue='Pclass', y='Survived', estimator=np.mean).set(xticklabels=["Male",
                                                                                                 "Female"])
plt.ylabel("Proportion of Survival")
plt.title("Survival by socio-economic status") 
sns.despine() 
```


![png](output_18_0.png)


The [webpage](https://www.kaggle.com/c/titanic/data) that contains the original data set suggests using the variable "Pclass" as a proxy for socioeconomic status. Wether or not this is a good proxy for socio economic status is of course debatable. The black lines on top of each bar represents 95 percent confidence intervals for each ticket class. Notice that most of them overlap. This means the differences in proportion of survival between ticket classes are not statistically significant at the 5 percent significance level. The only difference that is significant at the 5 percent level is for third class females. Most importantly, notice that the difference in survival for 1st class males and third class females is not statistically significant. These are important findings, but any causel inferences would be premature. Did third class females survive less than second or first class females *because* they were lower class, and lower class passengers were given lower priority in escaping? Or did this phenomenon occur because third class passengers simply located in a more deadly area during the accident? There is an obvious difference in survival for males and females, but its too early to leave SES(socioeconomic status) alone. Lets explore the idea of socioeconomic status a bit more by looking at the "Fare" variable...


```python
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
plt.figure(figsize=(10,8))

 
sns.violinplot(x = "Survived", y = "Fare", hue = "Female",  data = titanic_data).set(xticklabels=["Did not Survive",
                                                                                                 "Survived"])

plt.xlabel("Survival")
plt.title("Survival density by ticket fare")

sns.despine(left=True)
```


![png](output_20_0.png)


Here are some violin plots representing the probability density of survival based on ticket fare. The blue violins represent males and the green violins represents females. As you can see, the violins that represent the probability of not surviving appear much wider (more dense) around the line that represents a ticket fare of 0 dollars relative to the violins that represent the probability of surviving. In other words, the kernel density appears more spread out for the violins that represent passengers who survived. One obvious limitation to using ticket fare as a proxy for SES is that it is not clear from the description of the data wether or not the variable "Fare" represents the value paid by each passenger for an individual ticket, or the value paid for the whole ticket, including family members or children with nannies. The continious nature of the ticket fare data suggests that it may be the latter case, but there is not enough evidence to tell for sure.  


```python
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("ticks")
plt.figure(figsize=(10,8))
fare_bins = np.arange(0, 550, 10)
sns.distplot(titanic_data.loc[(titanic_data['Survived']==0),'Fare'], bins=fare_bins)
sns.distplot(titanic_data.loc[(titanic_data['Survived']==1),'Fare'], bins=fare_bins)
plt.title('Fare distribution among survival classes')
plt.ylabel('Frequency')
plt.legend(['Did not survive', 'Survived']);
sns.despine()
```


![png](output_22_0.png)


The plot above represents the Fare distribution across survival classes. Both distributions spike between 0 and 50, which is likely because most passengers payed a ticket fare in that range. It is certaintly a little difficult to see the overlapping bins, but the height of the bin representing ticket fares from 0-10 for the "Did not survive" distribution(in blue) is quite informative. Again, any conclusions here are premature. In order to fully understand the relationship between ticket fare and survivorship, we must apply strong statistical models to our data.


```python
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col # makes a nice table

def cse(regression):
    """Returns robust standard errors"""
    return regression.get_robustcov_results(cov_type='HAC',maxlags=1) 


reg1 = sm.ols(formula="Survived ~ Female + Fare", data=titanic_data).fit() 
reg2 = sm.ols(formula="Survived ~ Female + Fare + Age", data=titanic_data).fit() # All this was taken from the 
                                                                                           #statsmodels documentation
reg3 = sm.ols(formula="Survived ~ Female + Fare + Age + total_fam", data=titanic_data).fit()

try: # Me attempting to deal with a strange warning I get sometimes
    output = summary_col([cse(reg1),cse(reg2),cse(reg3)],stars=True) #applying the cse function 
    print output
except SettingWithCopyWarning as copy:
    print(copy) # I don't think it works, because whatever I am avoiding is a 'Warning' and not an exception
    

```

    /anaconda/lib/python2.7/site-packages/statsmodels/iolib/summary2.py:372: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      res.ix[:, 0][idx] = res.ix[:, 0][idx] + '*'
    /anaconda/lib/python2.7/site-packages/statsmodels/iolib/summary2.py:374: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      res.ix[:, 0][idx] = res.ix[:, 0][idx] + '*'
    /anaconda/lib/python2.7/site-packages/statsmodels/iolib/summary2.py:376: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      res.ix[:, 0][idx] = res.ix[:, 0][idx] + '*'


    
    =============================================
              Survived I Survived II Survived III
    ---------------------------------------------
    Age                  0.0000      -0.0006     
                         (0.0008)    (0.0008)    
    Fare      0.0016***  0.0016***   0.0019***   
              (0.0002)   (0.0002)    (0.0003)    
    Female    0.5227***  0.5228***   0.5447***   
              (0.0297)   (0.0297)    (0.0287)    
    Intercept 0.1480***  0.1474***   0.1834***   
              (0.0163)   (0.0246)    (0.0256)    
    total_fam                        -0.0411***  
                                     (0.0081)    
    =============================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


Here is where the meat of the presentation begins. The purpose of including three regressions in one table is to see how our independent variable of interest, Female, changes when we hold other variables constant. Specifically, we hold constant the variables that are likely correlated with survivorship. The main phenomon to notice is that when we hold Age, Fare, and total family on board constant, the relationship between female and survivorship appears *more* positive. However, we have certaintly not controlled for all related variables. One possible confounding variable that I did not include is the Pclass variable. Perhaps most women were in the first class and most men were in the second or third class (assuming this relationship between class and survivorship holds). Also notice no  statistically significant relationship between age and survivorship. The strange looking asterisks denote statistical significance, and the sandard errors are in parenthesis underneath the coefficients. The equation for the third regression can be written as follows:

$\widehat{Survived} = 0.1834 - 0.0006Age - 0.0411Fam + 0.0019Fare + 0.5447Female + U_i$ 

So a one year increase in age is associated with a 0.0006 decrease in probability of survival, holding all else constant. Also, a one unit increase in total family on board (the unit being one person) is associated with a 0.0411 decrease in probability of survival, all else equal. Moreover, a one dollar increase in ticket fare is associated with a 0.0019 increase in probability of survival, all else equal. Finally, on average, the probability for a female passenger to survive the titanic is 0.5447 higher than a male passenger, all else equal.   


```python
import statsmodels.formula.api as sm

result = sm.probit(formula="Survived ~ Fare + Female", data=titanic_data).fit() 

print result.summary() 
```

    Optimization terminated successfully.
             Current function value: 0.495530
             Iterations 6
                              Probit Regression Results                           
    ==============================================================================
    Dep. Variable:               Survived   No. Observations:                  891
    Model:                         Probit   Df Residuals:                      888
    Method:                           MLE   Df Model:                            2
    Date:                Tue, 29 Aug 2017   Pseudo R-squ.:                  0.2559
    Time:                        18:05:53   Log-Likelihood:                -441.52
    converged:                       True   LL-Null:                       -593.33
                                            LLR p-value:                 1.174e-66
    ==============================================================================
                     coef    std err          z      P>|z|      [95.0% Conf. Int.]
    ------------------------------------------------------------------------------
    Intercept     -1.0694      0.071    -15.083      0.000        -1.208    -0.930
    Fare           0.0067      0.001      5.238      0.000         0.004     0.009
    Female         1.4755      0.099     14.842      0.000         1.281     1.670
    ==============================================================================


Here we apply a probit regression model with Survived as the dependent variable, and Female along with Fare were included as regressors. The coefficients from the probit model are estimated by maximum likelihood rather than OLS. The formula for the above fitted probit model can be written as follows:

$\widehat{Pr(Survived | Age,Female)} = \phi (-1.0694 + 0.0067Fare + 1.4755Female)$

where $\phi$ is the cumulative standard normal distribution function, which you can find tabulated [here](http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf).
The probability of survival for a female who paid a $14 dollar entrance Fare* is about 69.15%, whereas the probability of survival for a male passenger who paid the same fare is about 16.35%. Additionally, when the fare is increased from 14 dollars to 31 dollars (the third quartile), the probability for a female to survive the titanic is about 72.91%. The probability of survival for a male who pays the same fare is 19.49%. So an increase from the 50% quartile to the 75% quartile in terms of Fare is associated with an increase of 3.76 percentage points for a female and an increase of 3.14 percentage points for a male. 	




\* Fun fact: Since 1912, the U.S. has experienced an average inflation rate of about 3.09% per year, which would mean that 14USD would be worth about 341.93USD in 2017. 

# Conclusions
================================================================

As mentioned earlier, while it may be tempting, one should not draw any conclusions from the results presented above. All findings presented so far are tentative, very tentative. However, to answer the question of what variables made a person more or less likely to survive the titanic, I provide my regression tables as evidence that being a female appears to be the variable that would make a passenger most likely to survive the Titanic, all else equal. But again, there may be other confounding variables that I did not include. From looking at the data, it seemed to me that passengers of the Titanic where probably putting females in the escape boats first, based on the idea of chivalry. However, I could find no evidence of this searching through the history of the Titanic. From what I've heard and read, the life boats were mostly a free for all. So why were women more likely to survive? This remains a mystery to me. 

Interestingly, I assumed that the ticket fare would be the variable that correlated most strongly with survivorship. There does appear to be a relationship, especially when looking at the figure titled "Fare distribution among survival classes", especially when fare is less than 50. However, when we increase the fare from the median value to the third quartile value in the probit model, the chances of surviving did not increase as much as I expected (3.14 percentage points for a male and 3.76 percentage points for a female). Because the relationship between ticket fare and survival seems to *decrease* as ticket fare increases, I should have included a variable Fare^2 in my probit regression model. However, it is still very clear that being female was the variable that made a passenger most likely to survive. 

#### Ideas for further exploration

I wanted to get the first project over with relatively quickly so I can move on with the course, but there were several methods I wish I had time to include in this presentation. For example, in addition to the probit model, I could have applied a logit model with the same data. Moreover, as I mentioned earlier, there could be a stronger relationship between survivorship and ticket fare when the ticket fare is low, so I could have included a variable for ticket fares squared in my regressions as well. Lastly, it would have been interesting to include the total fam variable in my probit regression.  


```python

```
