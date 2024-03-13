from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Iterable
import ir_datasets, gzip, joblib, json, os, re

class Preprocessing:

    def __init__ (self, dataset):

        self.dataset = dataset

    def tokenization_nltk (self, text: str) -> List[str]:

        return [ token for token in word_tokenize (text) ]

    def remove_noise_nltk (self, tokenized_text: List[str], preserve: List[str]) -> List[str]:

        return [ token for token in tokenized_text if re.match (r'[\w_-]', token) or token in preserve ]

    def remove_stopwords (self, tokenized_text: List[str], preserve: List[str]) -> List[str]:

        stop_words = set (stopwords.words ('english')) - set (preserve)
        return [ token for token in tokenized_text if token not in stop_words ]

    def morphological_reduction_nltk (self, tokenized_text: List[str], preserve: List[str], use_lemmatization = True) -> List[str]:

        lemmatizer = WordNetLemmatizer ()
        stemmer = PorterStemmer ()

        return [ token if token in preserve else lemmatizer.lemmatize (token) if use_lemmatization else stemmer.stem (token) for token in tokenized_text ]

    def preprocess_dataset (self):

        pass

    def preprocess_text (self, text, preserve = []):

        tokenized_text = self.tokenization_nltk (text)
        lowered_text = [ token if token in preserve else token.lower () for token in tokenized_text ]
        cleaned_text = self.remove_noise_nltk (lowered_text, preserve)
        filtered_text = self.remove_stopwords (cleaned_text, preserve)
        reduced_text = self.morphological_reduction_nltk (filtered_text, preserve)

        return reduced_text

class BooleanPreprocessing (Preprocessing):

    def __init__ (self, dataset):

        super ().__init__ (dataset)

    def build_vocabulary (self, texts):

        vocabulary = {}

        for text in texts:

            for token in text:

                if not vocabulary.get (token):

                    vocabulary[token] = True
        
        return list (vocabulary.keys ())

    def vector_representation (self, text, vocabulary):

        vector_repr = [ 1 if token in text else 0 for token in vocabulary ]
        return vector_repr

    def preprocess_dataset (self):

        processed = { doc.doc_id: self.preprocess_text (doc.text) for doc in self.dataset.docs_iter () }
        store = self.dataset.docs_store ()
        vocabulary = self.build_vocabulary ([ text for text in processed.values () ])

        for doc_id in processed.keys ():

            doc = store.get (doc_id)

            print (f"vectorizing {doc.doc_id}")

            processed[doc.doc_id] = {

                "title" : doc.title,
                "vector" : self.vector_representation (processed[doc.doc_id], vocabulary),
            }
        return processed, vocabulary

class VectorPreprocessing (Preprocessing):

    def __init__ (self, dataset):

        super ().__init__ (dataset)

    def preprocess_dataset (self, vocabulary: Iterable[str] | None):

        documents = [ doc for doc in self.dataset.docs_iter () ]
        processed = [ ' '.join (self.preprocess_text (doc.text)) for doc in documents ]

        vectorizer = TfidfVectorizer (vocabulary = vocabulary)
        processed = vectorizer.fit_transform (processed)

        X = processed.toarray ().tolist ()
        processed = { documents[i].doc_id: { 'title': documents[i].title, 'vector': X[i] } for i in range (len (X)) }
        return processed, vectorizer

if __name__ == '__main__':

    data_dir = 'data'

    if not os.path.exists (data_dir):

        os.makedirs (data_dir)

    dataset = ir_datasets.load ('beir/nfcorpus/test')

    print ('dataset loaded')

    boolean_preprocessor = BooleanPreprocessing (dataset)
    vector_preprocessor = VectorPreprocessing (dataset)

    processed, vocabulary = boolean_preprocessor.preprocess_dataset ()

    print ('boolean processing finished')

    with gzip.open  (os.path.join (data_dir, 'vocabulary.json.gz'), 'wt') as f:

        json.dump (vocabulary, f)

    with gzip.open (os.path.join (data_dir, 'boolean_corpus.json.gz'), 'wt') as f:

        json.dump (processed, f)

    print ('boolean processing written')

    preprocessor = VectorPreprocessing (dataset)
    processed, vectorizer = vector_preprocessor.preprocess_dataset (vocabulary)

    print ('vector processing finished')

    joblib.dump (vectorizer, os.path.join (data_dir, 'vectorizer.json.gz'), compress = 9)

    with gzip.open (os.path.join (data_dir, 'vector_corpus.json.gz'), 'wt') as f:

        json.dump (processed, f)

    print ('vector processing written')

    print ('processing completed!')
