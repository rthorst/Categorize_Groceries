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
streamlit.markdown("## Paste the list from Alexa here and press enter:")
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

    # Map departments to items.
    department_to_items = { department : [] for 
            department in predicted_departments}
    for department, item in zip(predicted_departments, shopping_list_items):
        department_to_items[department].append(item)

    # Output results as a printable list. 
    list_markdown = ""
    WHITESPACE = "&nbsp;"*5
    DIVIDER = WHITESPACE + "|" + WHITESPACE
    for department in sorted(department_to_items.keys()):
        
        # Add department as title.
        list_markdown += "#### {}\n".format(department.capitalize())
        
        # Add individual items in sorted order.
        items = sorted(department_to_items[department])
        last_item_idx = len(items) - 1
        for idx, item in enumerate(items):

            # Add item only. Then, for all items by last, 
            # add a pipe divider.
            list_markdown += "{}".format(item.capitalize())
            if idx != last_item_idx:
                list_markdown += DIVIDER

        # Add linebreak after last divider, otherwise, the next
        # header will not format properly.
        list_markdown += "\n"

    # Remove extra newline.
    list_markdown = list_markdown.rstrip()

    # Show grocery list.
    streamlit.markdown(list_markdown)
