import streamlit as st
import pandas as pd
import base64
import os
from PIL import Image
from indesearch import INDESearch

__author__ = 'Daniel Villanueva'
__email__ = '2144810@brunel.ac.uk'
__website__ = 'https://www.linkedin.com/in/danielvillanuevanunez/'
__copyright__ = 'Copyright 2022, Daniel Villanueva'

# Location of the tf-idf model.
tfidf_filename = "model/indecopi_resoluciones_tfidf.sav"
# Database
df_corpus = pd.read_csv("data/resoluciones_tfidf.csv")

def get_binary_file_downloader_html(file_location):
    with open(file_location, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(file_location)}">Download PDF</a>'
    return href


def main():
    st.title("Indecopi Search Engine 🔬")
    search = st.text_input("Enter search words:")
    if search:
        option = st.selectbox("How many documents do you want?", ("None",1,2,3,4,5,))
        if option in (1,2,3,4,5):
            # Instantiating the class.
            query_class = INDESearch(search,
                                     top_values = option,
                                     database = df_corpus)
            # Clean the query.
            clean_query = query_class.cleaning_query_tfidf(tfidf_filename)
            # Dataframe containing the results
            result = query_class.similarity(clean_query)
            for file_name in result["label"]:
                href = get_binary_file_downloader_html(file_name)
                st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
