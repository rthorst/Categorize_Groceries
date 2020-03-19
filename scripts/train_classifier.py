"""
Train a model to take the name of a product as input, e.g. all-seasons salt,
and classify the appropriate aisle, based on instacart data.

E.g. "artisan baguette" -> 3 (bakery)
"""
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle 

def train_model():

    """ 
    Load data, this consists of:
    -> Mapping of product names to department numbers.
    -> Mapping of department number to department name.
    """
    
    # Load mapping of products to department numbers.
    products_p = os.path.join("..", "data", "products.csv")
    print("read {}".format(products_p))
    products_df = pd.read_csv(products_p)
    print(products_df.columns)


    # Load mapping of department name -> department number.
    departments_p = os.path.join("..", "data", "departments.csv")
    print("read {}".format(departments_p))
    departments_df = pd.read_csv(departments_p)
    department_to_id = {}
    for dept, id in zip(departments_df.department, departments_df.department_id):
        department_to_id[dept] = id
    
    """ 
    Preprocess data:
    -> Vectorize product names
    -> Retain y- vector which is number of department.
        Replace all y=21 (missing) with y=2 (other)
    """

    # Vectorize product names.
    vectorizer = CountVectorizer(
        lowercase = True,
        max_features = 10000
    )
    print("vectorize product names")
    X = vectorizer.fit_transform(products_df.product_name)
    print("X shape {}".format(X.shape))

    # Get y vector, replacing all y=21 (missing) with y=2 (other)
    y = [2 if dept==21 else dept for dept in products_df.department_id]
    y = np.array(y)
    print("y shape {}".format(y.shape))

    # Train/test split.
    print("Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print("shapes : X train {} test {} , y train {} test {}".format(
            X_train.shape, X_test.shape, y_train.shape, y_test.shape
            ))
    """ 
    Train model
    """
    clf = LogisticRegression(
        penalty="l2", 
        verbose=1
    )
    print("fit model")
    clf.fit(X, y)

    """
    Evaluate model
    """
    report = classification_report(y_test, clf.predict(X_test))
    print(report)

    """
    Save results:
    -> model
    -> mapping of department # -> department name
    """
    
    objs = [clf, department_to_id, report, vectorizer]
    names = ["clf", "department_to_id", "classification_results", "vectorizer"]

    for obj, name in zip(objs, names):

        of_p = os.path.join("..", "results", "{}.p".format(name))
        pickle.dump(obj, open(of_p, "wb"))
        print("write {}".format(of_p))


class DepartmentClassifier:

    def __init__(self):

        """
        Load the needed objects, specifically:
        --> classifier  (map text -> department ID)
        --> vectorizer  (vectorize text of product name)
        --> mapping of department name to ID.
        """

        # Load vectorizer.
        vectorizer_p = os.path.join("..", "results", "vectorizer.p")
        self.vectorizer = pickle.load(open(vectorizer_p, "rb"))

        # Load classifier.
        clf_p = os.path.join("..", "results", "clf.p")
        self.clf = pickle.load(open(clf_p, "rb"))

        # Load mapping of department name to ID.
        name_to_id_p = os.path.join("..", "results", "department_to_id.p")
        self.department_name_to_id = pickle.load(open(name_to_id_p, "rb"))
        self.department_id_to_name = { v : k for k, v in self.department_name_to_id.items()}


    def get_department(self, product_name):
        
        # vectorize.
        X = self.vectorizer.transform([product_name])

        # predict department.
        department_int = self.clf.predict(X)[0]

        # get department name.
        department_name = self.department_id_to_name[department_int]
   
        return department_name

if __name__ == "__main__":
    #train_model()

    DC = DepartmentClassifier()
    items = ["cheese", "baguette", "spinach", "ice cream", "cookies"]
    for item in items:
        department = DC.get_department(item)
        print("{} -> {}".format(item, department))
