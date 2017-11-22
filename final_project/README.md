# Identify Persons of Interest from Enron

The file poi_id.py is the main file where you can find the new feature creation and the machine learning classifer. It first imports the raw dataset from final_project_dataset.pkl into a python dictionary. Next, some outliers in the dataset are dropped and a new feature is created.  Further, the dataset is split into training and testing data (the module feature_format.py is used to separate the features from the labels) and is used in a sklearn pipeline. Finally, the tester.py module is used to "dump" the classifier, dataset, and feature list into respective pickle files.

## How to use the program

Install the requirements

```
pip install -r requirements.txt
```

Run poi_id.py

```
python3 poi_id.py
```

poi_id.py will print out evaluation metrics such as precision, recall, cross-validation accuracy, and test-set score (it also includes a confusion matrix), but you should run tester.py for a more hands-on approach.

```
python3 tester.py
```

## Works cited

Thank you to Udacity for providing excellent starter code. Also, the mini-projects in the machine learning course really helped me get started.

**websites used**

* http://scikit-learn.org/

* https://stackoverflow.com/ 

* https://datascience.stackexchange.com (especially this post: https://datascience.stackexchange.com/questions/10773/how-does-selectkbest-work)

**Books**

* Introduction to Machine Learning with Python: A Guide for Data Scientists
By Andreas C. Müller, Sarah Guido
