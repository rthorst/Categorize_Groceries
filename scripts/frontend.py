import streamlit
import pickle
import re 
import numpy as np
import pandas as pd

def parse_shopping_list_into_individual_items(list_string):
    # Input e.g. "Shopping   1. Shredded cheese 2. Half and half .... 
    # ... Shared from the Amazon Alexa App
    
    # Remove unnecessary text from the right.
    list_string = list_string.replace(" Shared from the Amazon Alexa App", "")

    # Split the list into individual items by splitting on a number
    # and period, e.g. 1. or 13.
    pat = re.compile(r"\d+\.")
    items = re.split(pat, list_string)
    items = items[1:] # the 0th item is the list title: ignore it.

    # Remove unnecessary whitesppace and lowercase.
    items = [item.lower().lstrip().rstrip() for item in items]

    return items
    

# Preload the machine learning model and model objects.
clf = pickle.load(open("../results/clf.p", "rb"))
vectorizer = pickle.load(open("../results/vectorizer.p", "rb"))
department_to_id = pickle.load(open("../results/department_to_id.p", "rb"))
id_to_department = { k : v for v, k in department_to_id.items() }

# Provide a place to paste the shopping list.
streamlit.markdown("## Paste the list from Alexa here:")
list_text_from_alexa = streamlit.text_input("")

# If a shopping list is entered, parse it.
if len(list_text_from_alexa) > 1:

    # Parse the list into individual items.
    shopping_list_items = parse_shopping_list_into_individual_items(
            list_text_from_alexa)

    # Vectorize the shopping list.
    X = vectorizer.transform(shopping_list_items)

    # Classify grocery items.
    ypred = clf.predict(X)
    predicted_departments = [id_to_department[yi] for yi in ypred]

    # Output results as a sorted table.
    data = np.vstack([shopping_list_items, predicted_departments]).T
    colnames = ["Item", "Department"]
    df = pd.DataFrame(data=data, columns=colnames)
    df = df.sort_values(by = ["Department", "Item"])
    streamlit.dataframe(df)
