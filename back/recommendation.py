from nltk.corpus import wordnet
from back.preprocessing import Preprocessing
from sklearn.neighbors import NearestNeighbors
from typing import List
import numpy as np

class Recommendation:
    """
    A class for generating recommendations based on text similarity.

    Attributes
    ----------
    dataset : object
        The dataset object containing the documents to be used for recommendations.
    preprocessing : Preprocessing
        An instance of the Preprocessing class for text preprocessing.
    vectorizer : object
        The vectorizer object used to transform text into numerical vectors.
    knn : NearestNeighbors
        An instance of the NearestNeighbors class for finding the nearest neighbors.

    Methods
    -------
    __init__(dataset, vectorizer, vectorized_docs)
        Constructs all the necessary attributes for the Recommendation object.
    recommend(results)
        Generates recommendations based on the input results.
    """

    def __init__(self, dataset, vectorizer, vectorized_docs):
        """
        Constructs all the necessary attributes for the Recommendation object.

        Parameters
        ----------
        dataset : object
            The dataset object containing the documents to be used for recommendations.
        vectorizer : object
            The vectorizer object used to transform text into numerical vectors.
        vectorized_docs : list
            A list of vectorized documents.
        """
        self.dataset = dataset
        self.preprocessing = Preprocessing(dataset)
        self.vectorizer = vectorizer

        self.knn = NearestNeighbors(n_neighbors=10)
        self.knn.fit(np.array(vectorized_docs))

    def recommend(self, results: List[str]) -> List[str]:
        """
        Generates recommendations based on the input results.

        Parameters
        ----------
        results : List[str]
            A list of strings representing the input results.

        Returns
        -------
        List[str]
            A list of recommended documents based on the input results.
        """
        target = self.dataset.docs_store().get(results[0]).text
        target = self.preprocessing.preprocess_text(target)

        newtarget = []
        for word in target:
            synonyms = wordnet.synsets(word)
            synonyms = [lemma.name() for syn in synonyms for lemma in syn.lemmas()]
            newtarget.extend(synonyms)

        target = ' '.join(newtarget)
        target = self.vectorizer.transform([target])
        target = target.reshape(1, -1)

        distances, indices = self.knn.kneighbors(target)

        return indices[0]
