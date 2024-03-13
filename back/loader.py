from models import BooleanModel, ExtendedModel, LSIModel, Model
from back.recommendation import Recommendation
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import gzip, ir_datasets, joblib, json, os

class Loader:
    """
    A class to load and manage various models and data for information retrieval tasks.

    Attributes:
        dataset: The dataset to be used for the models.
        data_dir: The directory where the data files are stored.
        boolean_vocabulary_gz: The gzipped file name for the boolean vocabulary.
        vector_vectorizer_gz: The gzipped file name for the vectorizer.
    """

    boolean_vocabulary_gz = 'vocabulary.json.gz'
    vector_vectorizer_gz = 'vectorizer.json.gz'

    def __init__(self, dataset, data_dir):
        """
        Initializes the Loader with a dataset and a data directory.

        Args:
            dataset: The dataset to be used for the models.
            data_dir: The directory where the data files are stored.
        """
        self.dataset = dataset
        self.data_dir = data_dir

    def load_boolean(self, doc_vectors: list, vocabulary: List[str]) -> tuple[Model, dict]:
        """
        Loads a Boolean model with the given document vectors and vocabulary.

        Args:
            doc_vectors: A list of document vectors.
            vocabulary: A list of vocabulary words.

        Returns:
            A tuple containing the Boolean model and its arguments.
        """
        model = BooleanModel(self.dataset)
        arguments = {'doc_vectors': doc_vectors, 'vocabulary': vocabulary}
        return model, arguments

    def load_corpus(self, name: str) -> tuple[List[str], list]:
        """
        Loads a corpus from a gzipped JSON file.

        Args:
            name: The name of the gzipped JSON file.

        Returns:
            A tuple containing a list of document IDs and a list of document vectors.
        """
        with gzip.open(os.path.join(self.data_dir, name), 'rt') as f:
            print(f"Loading corpus from {os.path.join(self.data_dir, name)}")
            corpus = json.load(f)
            doc_ids = [key for key in corpus.keys()]
            doc_vectors = [doc["vector"] for doc in corpus.values()]
        return doc_ids, doc_vectors

    def load_extended(self, doc_vectors: list, vocabulary: List[str]) -> tuple[Model, dict]:
        """
        Loads an Extended model with the given document vectors and vocabulary.

        Args:
            doc_vectors: A list of document vectors.
            vocabulary: A list of vocabulary words.

        Returns:
            A tuple containing the Extended model and its arguments.
        """
        model = ExtendedModel(self.dataset)
        arguments = {'doc_vectors': doc_vectors, 'vocabulary': vocabulary}
        return model, arguments

    def load_lsi(self, doc_vectors: list, vectorizer: TfidfVectorizer) -> tuple[Model, dict]:
        """
        Loads an LSI model with the given document vectors and vectorizer.

        Args:
            doc_vectors: A list of document vectors.
            vectorizer: A TfidfVectorizer instance.

        Returns:
            A tuple containing the LSI model and its arguments.
        """
        model = LSIModel(self.dataset, doc_vectors, vectorizer)
        arguments = {'doc_vectors': doc_vectors}
        return model, arguments

    def load_relevant(self, name: str) -> List[str]:
        """
        Loads a list of relevant documents from a gzipped JSON file.

        Args:
            name: The name of the gzipped JSON file.

        Returns:
            A list of relevant document IDs.
        """
        with gzip.open(os.path.join(self.data_dir, name), 'rt') as f:
            relevant = json.load(f)
        return relevant

    def load_vocabulary(self):
        """
        Loads the vocabulary from a gzipped JSON file.

        Returns:
            A dictionary mapping tokens to their indices.
        """
        with gzip.open(os.path.join(self.data_dir, self.boolean_vocabulary_gz), 'rt') as f:
            print(f"Loading vocabulary from {os.path.join(self.data_dir, self.boolean_vocabulary_gz)}")
            array = json.load(f)
            vocabulary = {token: i for i, token in enumerate(array)}
        return vocabulary

    def load_vectorizer(self):
        """
        Loads the vectorizer from a gzipped file.

        Returns:
            The loaded vectorizer.
        """
        print(f"Loading vectorizer from {os.path.join(self.data_dir, self.vector_vectorizer_gz)}")
        return joblib.load(os.path.join(self.data_dir, self.vector_vectorizer_gz))

    def save_relevant(self, name: str, relevant: List[str]) -> None:
        """
        Saves a list of relevant documents to a gzipped JSON file.

        Args:
            name: The name of the gzipped JSON file.
            relevant: A list of relevant document IDs.
        """
        with gzip.open(os.path.join(self.data_dir, name), 'wt') as f:
            json.dump(relevant, f)

class DefaultLoader(Loader):
    """
    A default implementation of the Loader class with additional functionality for lazy loading.

    Attributes:
        boolean_corpus_gz: The gzipped file name for the boolean corpus.
        relevant_gz: The gzipped file name for the relevant documents.
        vector_corpus_gz: The gzipped file name for the vector corpus.
    """

    boolean_corpus_gz = 'boolean_corpus.json.gz'
    relevant_gz = 'relevant.json.gz'
    vector_corpus_gz = 'vector_corpus.json.gz'

    def __init__(self, lazy=True):
        """
        Initializes the DefaultLoader with optional lazy loading.

        Args:
            lazy: A boolean indicating whether to load data lazily.
        """
        dataset = ir_datasets.load("beir/nfcorpus/test")
        data_dir = 'data/'
        super().__init__(dataset, data_dir)
        self.boolean_model = None
        self.boolean_vectors = None
        self.extended_model = None
        self.extended_vectors = None
        self.lsi_model = None
        self.recommender = None
        self.vocabulary = None
        self.vectorizer = None
        if not lazy:
            self.load_boolean()
            self.load_extended()
            self.load_recommender()
            self.load_vocabulary()
            self.load_vectorizer()

    def load_boolean (self):

        if not self.boolean_model:

            self.load_boolean_corpora ()
            self.load_vocabulary ()

            model, arguments = super ().load_boolean (self.boolean_vectors, self.vocabulary)

            self.boolean_model = model
            self.boolean_arguments = arguments

    def load_boolean_corpora (self):

        if not self.boolean_vectors:

            doc_ids, doc_vectors = self.load_corpus (self.boolean_corpus_gz)

            self.boolean_ids = doc_ids
            self.boolean_vectors = doc_vectors

    def load_extended (self):

        if not self.extended_model:

            self.load_extended_corpora ()
            self.load_vocabulary ()

            model, arguments = super ().load_extended (self.extended_vectors, self.vocabulary)

            self.extended_model = model
            self.extended_arguments = arguments

    def load_extended_corpora (self):

        if not self.extended_vectors:

            doc_ids, doc_vectors = self.load_corpus (self.vector_corpus_gz)

            self.extended_ids = doc_ids
            self.extended_vectors = doc_vectors

            self.vector_vectors = { doc_id: vector for doc_id, vector in zip (doc_ids, doc_vectors) }

    def load_lsi (self):

        if not self.lsi_model:

            self.load_extended_corpora ()
            self.load_vectorizer ()

            model, arguments = super ().load_lsi (self.extended_vectors, self.vectorizer)

            self.lsi_model = model
            self.lsi_arguments = arguments

    def load_recommender (self):

        if not self.recommender:

            self.load_extended_corpora ()
            self.load_vectorizer ()

            self.recommender = Recommendation (self.dataset, self.vectorizer, self.extended_vectors)

    def load_vocabulary (self):

        if not self.vocabulary:

            self.vocabulary = super ().load_vocabulary ()

    def load_vectorizer (self):

        if not self.vectorizer:

            self.vectorizer = super ().load_vectorizer ()

    def get_doc (self, doc_id: str):

        return self.dataset.docs_store ().get (doc_id)

    def get_relevant (self):

        return super ().load_relevant (self.relevant_gz)

    def recommend (self, result: str) -> List[str]:

        self.load_recommender ()

        matches = self.recommender.recommend ([result])
        matches = [ self.extended_ids[i] for i in matches ]
        return matches

    def save_relevant (self, relevant: List[str]):

        return super ().save_relevant (self.relevant_gz, relevant)

    def search_boolean (self, query: str) -> List[str]:

        self.load_boolean ()

        query = self.boolean_model.create_query (query)
        matches = self.boolean_model.match_vectors (query, relevant = self.get_relevant (), **self.boolean_arguments)
        matches = [ self.boolean_ids[i] for i in matches ]
        return matches

    def search_extended (self, query: str) -> List[str]:

        self.load_extended ()

        query = self.extended_model.create_query (query)
        matches = self.extended_model.match_vectors (query, relevant = self.get_relevant (), **self.extended_arguments)
        matches = [ self.extended_ids[i] for i in matches ]
        return matches

    def search_lsi (self, query: str) -> List[str]:

        self.load_lsi ()

        relevant = self.get_relevant ()
        irrelevant = set (self.extended_ids) - set (relevant)

        relevant = [ self.vector_vectors[doc_id] for doc_id in relevant ]
        irrelevant = [ self.vector_vectors[doc_id] for doc_id in irrelevant ]

        query = self.lsi_model.create_query (query)
        matches = self.lsi_model.match_vectors (query, relevant = relevant, irrelevant = irrelevant, **self.lsi_arguments)
        matches = [ self.extended_ids[i] for i in matches ]
        return matches
