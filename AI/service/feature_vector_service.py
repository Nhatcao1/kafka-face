import abc


class FeatureVectorService(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search(self, query_vectors, top_k=1, collection_name=None, partition_tags=None):
        """
        search n nearest vector from the list of vector.
        :param collection_name: name of vector collection
        :param query_vectors: list of vectors need to be searched
        :param top_k: total nearest vector will be return
        :param partition_tags: partition tag list
        :return: nearest vector ids matrix and distance matrix
        """
        ...

    @abc.abstractmethod
    def insert_vectors(self, vectors, ids=None, collection_name=None, partition_tag=None):
        """
        insert vectors to search vector engine
        :param vectors: vectors list[list[float]]. example vectors: [[1.2345],[1.2345]]
        :param ids: list of id. If set None, the system will auto generate the ids
        :param collection_name: name of vector collection
        :param partition_tag: partition tag
        :return: list of id
        """

        ...

    @abc.abstractmethod
    def remove_vectors(self, ids, collection_name=None):
        """
        remove vectors from search vector engine
        :param ids: list of id
        :param collection_name: name of vector collection
        :return: boolean
        """
        ...

    @abc.abstractmethod
    def create_collection(self, collection_name):
        """
        Create a collection
        Args:
            collection_name: (str) name of the collection
        Raises:
            Exception: milvus exception
        """
        ...

    @abc.abstractmethod
    def remove_collection(self, collection_name):
        """
        Remove collection using name
        Args:
            collection_name:
        Returns:
            status: (boolean) whether the operation succeeded (including collection_name does not exist)
        Raises:
            Exception: milvus exception
        """
        ...

    @abc.abstractmethod
    def count_entities(self, collection_name=None):
        """
        Count entities in an existed collection
        Args:
            collection_name: (str) name of the collection
        Returns:
            count: (int) number of entities in the collection
        Raises:
            InternalException
        """
        ...
