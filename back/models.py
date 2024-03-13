from collections import namedtuple
from preprocessing import Preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sympy.core import Symbol
from sympy.logic import And, Or
from typing import Dict, List, Union
import math, sympy
import numpy as np

BooleanQuery = namedtuple('BooleanQuery', ['query', 'tokens'])

class Model:
    """
    Base class for models that can create queries and match vectors.
    """

    def __init__(self, dataset):
        """
        Initialize the model with a dataset.

        :param dataset: The dataset to use for preprocessing.
        """
        self.preprocess = Preprocessing(dataset)

    def create_query(self, query, **kwargs):
        """
        Create a query from the given text.

        :param query: The text to create a query from.
        :return: A BooleanQuery object.
        """
        pass

    def match_vectors(self, query, **kwargs):
        """
        Match vectors based on the given query.

        :param query: The query to match vectors against.
        :return: A list of matching document IDs.
        """
        pass

class BooleanModel(Model):
    """
    A model that uses boolean logic to match queries against documents.
    """

    def __init__(self, dataset):
        """
        Initialize the BooleanModel with a dataset.

        :param dataset: The dataset to use for preprocessing.
        """
        super().__init__(dataset)

    def debug_query(self, text, preserve=[]):
        """
        Debug a query by tokenizing, lowering, cleaning, filtering, and reducing the text.

        :param text: The text to debug.
        :param preserve: A list of tokens to preserve.
        """
        tokenized_text = self.preprocess.tokenization_nltk(text)
        lowered_text = [token if token in preserve else token.lower() for token in tokenized_text]
        cleaned_text = self.preprocess.remove_noise_nltk(lowered_text, preserve)
        filtered_text = self.preprocess.remove_stopwords(cleaned_text, preserve)
        reduced_text = self.preprocess.morphological_reduction_nltk(filtered_text)

        print('debug query:')
        print(f'tokenized_text: {tokenized_text}')
        print(f'lowered_text: {lowered_text}')
        print(f'cleaned_text: {cleaned_text}')
        print(f'filtered_text: {filtered_text}')
        print(f'reduced_text: {reduced_text}')

    def create_query(self, query, **kwargs) -> BooleanQuery:
        """
        Create a BooleanQuery from the given query text.

        :param query: The query text.
        :return: A BooleanQuery object.
        """
        protected = ['AND', 'NOT', 'OR']
        tokenized_query = self.preprocess.preprocess_text(query, protected)
        boolean_query = tokenized_query.copy()

        for i in range(len(boolean_query)):
            if boolean_query[i] not in protected:
                boolean_query[i] = f'_{i}'
            if boolean_query[i] == "AND":
                boolean_query[i] = "&"
            elif boolean_query[i] == "OR":
                boolean_query[i] = "|"
            elif boolean_query[i] == "NOT":
                boolean_query[i] = "& ~"
            elif (i > 0 and boolean_query[i] not in ['&', '|', '& ~', '& ('] and boolean_query[i - 1] not in ['&', '|', '& ~', '& (']):
                boolean_query[i] = "& " + boolean_query[i]

        if len(boolean_query) == 0:
            raise Exception('empty query')

        processed_query = ' '.join(boolean_query)
        processed_query = sympy.sympify(processed_query, evaluate=False)

        return BooleanQuery(query=sympy.to_dnf(processed_query, simplify=True, force=True), tokens=tokenized_query)

    def sim(self, query: And | Or | Symbol, doc: List[int], tokens: List[str], vocabulary: Dict[str, int]) -> int:
        """
        Calculate the similarity between a query and a document.

        :param query: The query to compare.
        :param doc: The document vector.
        :param tokens: The tokens of the query.
        :param vocabulary: The vocabulary mapping.
        :return: The similarity score.
        """
        if type(query) == Symbol:
            return self.simclaus(query, doc, tokens, vocabulary)
        elif query.args is not None:
            for claus in query.args:
                if self.simclaus(claus, doc, tokens, vocabulary) > 0:
                    return 1
        return 0

    def simclaus(self, claus: And | Symbol, doc: List[int], tokens: List[str], vocabulary: Dict[str, int]) -> int:
        """
        Calculate the similarity of a clause within a query.

        :param claus: The clause to compare.
        :param doc: The document vector.
        :param tokens: The tokens of the query.
        :param vocabulary: The vocabulary mapping.
        :return: The similarity score.
        """
        atoms = [claus] if type(claus) == Symbol else claus.atoms()

        for atom in atoms:
            if self.simatom(atom, doc, tokens, vocabulary) == 0:
                return 0
        return len(atoms) > 0

    def simatom(self, atom: Symbol, doc: List[int], tokens: List[str], vocabulary: Dict[str, int]) -> int:
        """
        Calculate the similarity of an atom within a clause.

        :param atom: The atom to compare.
        :param doc: The document vector.
        :param tokens: The tokens of the query.
        :param vocabulary: The vocabulary mapping.
        :return: The similarity score.
        """
        name = tokens[int(atom.name[1:])]
        word = vocabulary.get(name)

        if word is None and not atom.is_Not:
            return 0
        elif word is None and atom.is_Not:
            return 1
        else:
            return doc[word] if not atom.is_Not else 1 - doc[word]
        return 0

    def match_vectors(self, boolean_query: BooleanQuery, **kwargs) -> List[int]:
        """
        Match vectors based on the given BooleanQuery.

        :param boolean_query: The BooleanQuery to match vectors against.
        :return: A list of matching document IDs.
        """
        matching_documents = []
        doc_vectors = kwargs.get('doc_vectors')
        relevant = kwargs.get('relevant', [])
        vocabulary = kwargs.get('vocabulary')

        query = boolean_query.query
        tokens = boolean_query.tokens

        for doc_id, doc_vector in enumerate(doc_vectors):
            docsim = self.sim(query, doc_vector, tokens, vocabulary)

            if docsim > 0:
                docsim = docsim if doc_id not in relevant else docsim * 2
                matching_documents.append((docsim, doc_id))

        matching_documents = sorted(matching_documents, key=lambda x: x[0])

        return [t[1] for t in matching_documents]

class ExtendedModel(BooleanModel):
    """
    An extended model that inherits from BooleanModel.
    
    This class provides additional functionality for similarity calculation
    between queries and documents.
    """

    def __init__(self, dataset):
        """
        Initialize the ExtendedModel with a given dataset.
        
        Parameters:
        - dataset: The dataset to be used for similarity calculations.
        """
        super().__init__(dataset)

    def sim(self, query: Union[And, Or, Symbol], doc: List[int], tokens: List[str], vocabulary: Dict[str, int]) -> float:
        """
        Calculate the similarity between a query and a document.
        
        Parameters:
        - query: The query to be compared with the document.
        - doc: A list of integers representing the document.
        - tokens: A list of strings representing the tokens in the document.
        - vocabulary: A dictionary mapping tokens to their integer IDs.
        
        Returns:
        - A float representing the similarity between the query and the document.
        """
        if type(query) == Symbol:
            return self.simclaus(query, doc, tokens, vocabulary)
        elif query.args is not None:
            weight = 0
            nargs = len(query.args)
            for claus in query.args:
                claussim = self.simclaus(claus, doc, tokens, vocabulary)
                weight += claussim ** 2
            return math.sqrt(weight / nargs)
        return 0

    def simclaus(self, claus: Union[And, Or, Symbol], doc: List[int], tokens: List[str], vocabulary: Dict[str, int]) -> float:
        """
        Calculate the similarity of a clause within a query.
        
        Parameters:
        - claus: The clause to be compared with the document.
        - doc: A list of integers representing the document.
        - tokens: A list of strings representing the tokens in the document.
        - vocabulary: A dictionary mapping tokens to their integer IDs.
        
        Returns:
        - A float representing the similarity of the clause within the query.
        """
        atoms = [claus] if type(claus) == Symbol else claus.atoms()
        natoms = len(atoms)
        weight = 0
        for atom in atoms:
            name = tokens[int(atom.name[1:])]
            word_id = vocabulary.get(name)
            if word_id is None:
                return 0
            else:
                if atom.is_Not:
                    weight += doc[word_id] ** 2
                else:
                    weight += (1 - doc[word_id]) ** 2
        return 1 - math.sqrt(weight / natoms)

def matrix_from_vectors(doc_vectors):
    """
    Create a matrix from a list of document vectors.
    
    Parameters:
    - doc_vectors: A list of document vectors.
    
    Returns:
    - A numpy array representing the matrix.
    """
    matrix = np.zeros((len(doc_vectors), len(doc_vectors[0])))
    for i, vector in enumerate(doc_vectors):
        matrix[i] = vector
    return matrix

def reduce_matrix_from(matrix, n_components):
    """
    Reduce the dimensionality of a matrix using TruncatedSVD.
    
    Parameters:
    - matrix: The input matrix to be reduced.
    - n_components: The number of components to keep.
    
    Returns:
    - A tuple containing the reduced matrix and the TruncatedSVD object.
    """
    svd = TruncatedSVD(n_components=n_components)
    lsi = svd.fit_transform(matrix)
    return lsi, svd

class LSIModel(Model):
    """
    A Latent Semantic Indexing (LSI) model for document similarity.
    
    This class uses LSI to transform documents into a lower-dimensional space
    for similarity calculations.
    """

    def __init__(self, dataset, doc_vectors, vectorizer, n_components=100):
        """
        Initialize the LSIModel with a dataset, document vectors, and a vectorizer.
        
        Parameters:
        - dataset: The dataset to be used for similarity calculations.
        - doc_vectors: A list of document vectors.
        - vectorizer: A vectorizer object for transforming text into vectors.
        - n_components: The number of components to keep in the LSI model (default 100).
        """
        super().__init__(dataset)
        self.matrix = matrix_from_vectors(doc_vectors)
        self.vectorizer = vectorizer
        lsi, svd = reduce_matrix_from(self.matrix, n_components)
        self.lsi = lsi
        self.svd = svd

    def create_query(self, query, **kwargs):
        """
        Create a query vector from a text query.
        
        Parameters:
        - query: The text query to be transformed into a vector.
        
        Returns:
        - A numpy array representing the query vector.
        """
        tokens = self.preprocess.preprocess_text(query)
        matrix = self.vectorizer.transform([' '.join(tokens)])
        vector = matrix.toarray()[0]
        return vector

    def match_vectors(self, query, **kwargs):
        """
        Match a query vector against the LSI model to find similar documents.
        
        Parameters:
        - query: The query vector to be matched.
        
        Returns:
        - A list of indices of documents that are similar to the query.
        """
        irrelevant = kwargs.get('irrelevant', [])
        relevant = kwargs.get('relevant', [])
        threshold = kwargs.get('threshold', 0.5)
        if len(relevant) > 0 and len(irrelevant) > 0:
            query = self.rocchio(query, relevant, irrelevant)
        lsi_vector = self.svd.transform([query])
        similarities = cosine_similarity(lsi_vector, self.lsi)
        indices = np.where(similarities[0] > threshold)[0]
        return indices

    def rocchio(self, query, relevant, irrelevant, alpha=1, beta=0.8, gamma=0.1):
        """
        Apply the Rocchio algorithm to adjust the query vector.
        
        Parameters:
        - query: The original query vector.
        - relevant: A list of vectors representing relevant documents.
        - irrelevant: A list of vectors representing irrelevant documents.
        - alpha: The weight for the original query vector (default 1).
        - beta: The weight for the relevant documents (default 0.8).
        - gamma: The weight for the irrelevant documents (default 0.1).
        
        Returns:
        - A numpy array representing the adjusted query vector.
        """
        irrelevant = np.mean(irrelevant, axis=0)
        relevant = np.mean(relevant, axis=0)
        return alpha * query + beta * relevant - gamma * irrelevant
