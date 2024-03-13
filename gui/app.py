from collections import namedtuple
from emoji import emojize
from st_aggrid import AgGrid, AgGridReturn, GridOptionsBuilder, GridUpdateMode
import json, os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Adjust the current directory and append back directories to the system path
current_directory = os.path.dirname(os.path.abspath(__file__))
back_directory = os.path.join(current_directory, '..')
sys.path.append(back_directory)
current_directory = os.path.dirname(os.path.abspath(__file__))
back_directory = os.path.join(current_directory, '..', 'back')
sys.path.append(back_directory)

# Import the DefaultLoader from the back module
from back.loader import DefaultLoader

# Define namedtuples for Match and Matches
Match = namedtuple('Match', ['id', 'title', 'text'])
Matches = namedtuple('Matches', ['boolean', 'extended', 'lsi'])

@st.cache_resource
def get_loader():
    """
    Initializes and returns a DefaultLoader instance with lazy loading disabled.
    
    Returns:
        DefaultLoader: An instance of DefaultLoader with lazy loading disabled.
    """
    return DefaultLoader(lazy=False)

@st.cache_data
def get_matches(query):
    """
    Retrieves matches for a given query using different search methods.
    
    Args:
        query (str): The search query.
    
    Returns:
        Matches: A namedtuple containing boolean, extended, and lsi matches.
    """
    loader = get_loader()
    boolean = loader.search_boolean(query)
    extended = loader.search_extended(query)
    lsi = loader.search_lsi(query)
    return Matches(boolean=boolean, extended=extended, lsi=lsi)

@st.cache_data
def get_recommendations(query):
    """
    Generates recommendations based on the extended matches of a given query.
    
    Args:
        query (str): The search query.
    
    Returns:
        list: A list of recommended matches.
    """
    loader = get_loader()
    matches = get_matches(query)
    extended_matches = matches.extended
    recommendations = [] if len(extended_matches) == 0 else loader.recommend(extended_matches[0])
    return recommendations

@st.cache_data
def get_relevant():
    """
    Retrieves relevant documents.
    
    Returns:
        list: A list of relevant document IDs.
    """
    loader = get_loader()
    try:
        return loader.get_relevant()
    except FileNotFoundError:
        return []

# Initialize the loader
get_loader()

# Streamlit UI setup
st.title('Welcome to our SRI')
col1, col2 = st.columns([4, 1])
query = col1.text_input('query', key='query_input', label_visibility='hidden', placeholder='Enter your query here')

if col2.button(emojize(":mag_right: Search")):
    loader = get_loader()
    matches = get_matches(query)
    recommendations = get_recommendations(query)

    session = st.session_state
    session['boolean_matches'] = [Match(loader.get_doc(doc_id).doc_id, loader.get_doc(doc_id).title, loader.get_doc(doc_id).text) for doc_id in matches.boolean]
    session['extended_matches'] = [Match(loader.get_doc(doc_id).doc_id, loader.get_doc(doc_id).title, loader.get_doc(doc_id).text) for doc_id in matches.extended]
    session['lsi_matches'] = [Match(loader.get_doc(doc_id).doc_id, loader.get_doc(doc_id).title, loader.get_doc(doc_id).text) for doc_id in matches.lsi]
    session['recommendations'] = [Match(loader.get_doc(doc_id).doc_id, loader.get_doc(doc_id).title, loader.get_doc(doc_id).text) for doc_id in recommendations]

def grid(df: pd.DataFrame, key: str = 'grid', checkable=True) -> AgGridReturn:
    """
    Creates a grid view of a DataFrame using Streamlit AgGrid.
    
    Args:
        df (pd.DataFrame): The DataFrame to display.
        key (str, optional): The key for the grid. Defaults to 'grid'.
        checkable (bool, optional): Whether the grid rows should be checkable. Defaults to True.
    
    Returns:
        AgGridReturn: The AgGrid instance.
    """
    gd = GridOptionsBuilder.from_dataframe(df)
    if checkable:
        relevant = set(get_relevant())

        #
        # Bug time: selected should be a List[int] but streamlit-aggrid has a bug on this
        # and selectes whatever rows it's hearts desires instead of what selected contains
        #
        # Workaround: https://github.com/PablocFonseca/streamlit-aggrid/issues/207
        #

        selected = {row.Index: True for row in df.itertuples() if row.ID in relevant}
        gd.configure_selection(pre_selected_rows=selected, selection_mode='multiple', use_checkbox=True)
    table = AgGrid(df, height=250, gridOptions=gd.build(), key=key, update_mode=GridUpdateMode.SELECTION_CHANGED)
    return table

boolean_matches = st.session_state.get('boolean_matches', [])
extended_matches = st.session_state.get('extended_matches', [])
lsi_matches = st.session_state.get('lsi_matches', [])
recommendations = st.session_state.get('recommendations', [])

grid1 = grid(pd.DataFrame(boolean_matches, columns=['ID', 'Título', 'Texto']), 'boolean_grid')
grid2 = grid(pd.DataFrame(extended_matches, columns=['ID', 'Título', 'Texto']), 'extended_grid')
grid3 = grid(pd.DataFrame(lsi_matches, columns=['ID', 'Título', 'Texto']), 'lsi_grid')
grid4 = grid(pd.DataFrame(recommendations, columns=['ID', 'Título', 'Texto']), 'recommendations_grid', False)

if st.button(emojize(":thumbsup: Mark as relevant")):
    selected1 = [row['ID'] for row in grid1.selected_rows]
    selected2 = [row['ID'] for row in grid2.selected_rows]
    selected3 = [row['ID'] for row in grid3.selected_rows]

    unselected1 = [match.id for match in boolean_matches if match.id not in selected1]
    unselected2 = [match.id for match in extended_matches if match.id not in selected2]
    unselected3 = [match.id for match in lsi_matches if match.id not in selected3]

    relevant = set(get_relevant())
    selected = set.union(set(selected1), set(selected2), set(selected3))
    unselected = set.union(set(unselected1), set(unselected2), set(unselected3))

    relevant = relevant - unselected
    relevant = set.union(relevant, selected)

    get_loader().save_relevant(list(relevant))
    get_relevant.clear()
