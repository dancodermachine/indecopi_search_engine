# Indecopi Search Engine: Project Overiew

![Search Engine GIF](demo_gif.gif)

**Problem:**
1. Every time there is a legal problem, a lawyer needs to search the Indecopi database for jurisprudence. However, the current search engine is not accurate. *Example: You received a case in which they denied the registration of a brand because it went against morality and good customs. Then a lawyer needs to know if this similar case happened before and know if the judge gave his veridict in favour or against the case. Consequently, he can use this past case to support his case.*

2. Students currently need to search for jurisprudence to successfully complete their research assigments or thesis. Again, the search engine is not accurate.

To see existant search engine click [here](https://servicio.indecopi.gob.pe/buscadorResoluciones/).

**Solution:**
I built a search engine which searches for documents calculating cosine similarity between the query and the PDF documents. 

**Definitions:**
1. Jurisprudence: Set of sentences, decisions or judgments issued by the courts of justice or government authorities.

## Requirements

* Python 3.10.2 (64-bit)
* Pandas 1.4.0
* NumPy 1.22.1
* pdfplumber 0.6.0
* Pickle 
* NLTK 3.6.7
* Scikit-Learn 1.0.2
* Gensim 4.1.2 (for Doc2Vec).
* Streamlit 1.4.0 (for displaying website).

## Installation
Go to the folder in your local computer and use the comand below:

`streamlit run search_app.py`

## Data Cleaning

* Text was extracted from each document.
* Punctuations and stopwords were removed from the text. 
* Each word was stemmed.
* Each document was vectorized using tf-idf.
* A matrix was created were each row is a document and its vectorized form. This was saved in a csv file.

## Project Content

* `data` folder contains `resoluciones_tfidf.csv` and `resoluciones_doc2vec.csv` files. The first file contains the vectorized form of the PDF documents using the TF-IDF model. The second file contains the vectorized form of the PDF documents using the Doc2Vec model.
* `model` folder contains `indecopi_resoluciones_tfidf.sav` and `indecopi_resoluciones_doc2vec.model` files. The first file is the model to convert text into a vector form using tfidf. The second files is the model to convert text into a vector form using Doc2Vec.
* `resoluciones` folder contrains 30 PDF files. 
* The `indesearch.py` file contains the code to clean the query and search for the files.
* The `search_app.py` files contains the user interface of the search engine. 
* `rank_tfidf.ipynb` file contains jupyter notebook containing the data cleaning, tf-idf model building, and the example on how does the `indesearch.py` file works. 
* The `rank_doc2vec.ipynb` file contains jupyter notebook containing the data cleaning, Doc2Vec model building, and the example on how does the `indesearch.py` file works. However, this model does not output accurate results 


## Model Measurements

To measure the model, it needs to be deployed and a likeable button needs to be placed in order to count how many people feel that their searches are useful. 

## Future work

1. The `pdf_to_text` function in the `rank_tfidf.ipynb` jupyter notebook needs to be optimized. It takes a lot of time to run. Since this is a nested loop the big O notation for this function is O(n^2).
2. `rank_doc2vec.ipynb` file needs a lot of improvement. Currently the model that uses Doc2Vec is not giving food results. Maybe more documents are needed to train the model. 
3. ElasticSearch is a well-known tool used in search engines. Therefore, this tool could be added to improve the model. Currently I do not have the knowledge of this tool. 


## Contact

For any queries click [here](https://www.linkedin.com/in/danielvillanuevanunez/) to contact me through LinkedIn. 

## Acknowledgement

Special thanks to Claudia Ávila and Honorio Apaza for providing me the documents.







