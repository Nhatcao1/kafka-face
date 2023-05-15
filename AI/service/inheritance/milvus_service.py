from log import logger
from milvus import Milvus, IndexType, MetricType
from milvus.client.exceptions import NotConnectError
import time
import datetime

from core.config import get_app_settings
from error import error_code
from error.customized_exception import BadRequestException, InternalException
from service.feature_vector_service import FeatureVectorService


class MilvusService(FeatureVectorService):
    def __init__(self):
        self.settings = get_app_settings()
        self.milvus = self.get_milvus_client()
        while not self.milvus:
            self.milvus = self.get_milvus_client()

        self.search_param = {
            "nprobe": 32
        }

    def get_milvus_client(self):
        try:
            milvus = Milvus(host=self.settings.milvus_host, port=self.settings.milvus_port)
            return milvus
        except NotConnectError:
            logger.info("Milvus connection refused")
            time.sleep(5)
            return None

    def count_entities(self, collection_name=None):
        if collection_name is None:
            raise InternalException("Collection name is empty")
        if self.milvus:
            status, count = self.milvus.count_entities(collection_name=collection_name)
            if not status.OK():
                raise InternalException(status.message)
        else:
            self.milvus = self.get_milvus_client()
            return 0
        return count

    def create_collection(self, collection_name):
        if not self.milvus:
            self.milvus = self.get_milvus_client()
        else:
            status, ok = self.milvus.has_collection(collection_name)
            if not ok:
                param = {
                    'collection_name': collection_name,
                    'index_file_size': 1024,
                    'dimension': self.settings.feature_vector_dimension,
                    'metric_type': MetricType.L2
                }

                self.milvus.create_collection(param)

                # present collection info
                _, info = self.milvus.get_collection_info(collection_name)
                logger.info(info)

                # Recommend to use IVF_FLAT if expected vector to be large enough, like 10000
                # Else stick with FLAT by default, to gain 100% Recall
                if self.settings.milvus_index == "ivf_flat":
                    ivf_param = {'nlist': 4096}
                    _ = self.milvus.create_index(collection_name, IndexType.IVF_FLAT, ivf_param)
                # describe index, get information of index
                status = self.milvus.get_index_info(collection_name)
                logger.info(status)

    def remove_collection(self, collection_name):
        status, exist = self.milvus.has_collection(collection_name)
        if not exist:
            logger.info("Milvus | collection: {} does not exist".format(collection_name))
            return True
        else:
            status = self.milvus.drop_collection(collection_name)
            logger.info("Milvus | collection: {} is removed".format(collection_name))
            return status.OK()

    def create_partition(self, collection_name, partition_tag):
        status, ok = self.milvus.has_partition(collection_name, partition_tag)
        if not ok:
            logger.info("partition name: {} not found. Create new one.".format(partition_tag))
            self.milvus.create_partition(collection_name, partition_tag=partition_tag)

    def get_index_info(self, collection_name: str):
        if collection_name is None:
            raise InternalException("Collection name is empty")
        status, info = self.milvus.get_collection_stats(collection_name)
        logger.info("Total amount of vectors in collection {} is {}".format(collection_name, info["row_count"]))

    def search(self, query_vectors, top_k=1, collection_name=None, partition_tags=None):
        """
        search n nearest vector from the list of vector.
        :param collection_name: name of vector collection
        :param query_vectors: list of vectors need to be searched
        :param top_k: total nearest vector will be return
        :param partition_tags: partition tag list
        :return: nearest vector ids matrix and distance matrix
        """
        if collection_name is None:
            raise InternalException("Collection name is empty")
        status, ok = self.milvus.has_collection(collection_name)
        if not ok:
            logger.info("collection not found")
            raise BadRequestException(error_code.FACESET_NOTFOUND, "faceset_token", collection_name)
        param = {
            'collection_name': collection_name,
            'query_records': query_vectors,
            'top_k': top_k,
            'params': self.search_param,
        }

        if partition_tags is not None:
            querry_partition_tags = []
            for partition_tag in partition_tags:
                status, ok = self.milvus.has_partition(collection_name, partition_tag)
                if ok:
                    querry_partition_tags.append(partition_tag)
            param["partition_tags"] = querry_partition_tags

        start = datetime.datetime.now()
        status, results = self.milvus.search(**param)
        end = datetime.datetime.now()
        logger.debug("Milvus search {} vector(s), top_k={}. Search time: {}ms".format(len(query_vectors),
                                                                                      top_k,
                                                                                      (end - start).microseconds / 1000))
        if status.OK():
            # indicate search result
            return results.id_array, results.distance_array
        else:
            logger.info("Search failed. {}".format(status))

    def insert_vector(self, vector, face_id=None, collection_name=None, partition_tag=None):
        """
        insert a vector to search vector engine
        :param vector: vector list[float]. example vectors: [1.2345, 1.2345]
        :param face_id: id of face. If set None, the system will auto generate the id
        :param collection_name: name of vector collection
        :param partition_tag: partition tag
        :return: list of face_id
        """
        # self.milvus.
        if face_id is None:
            return self.insert_vectors([vector], collection_name=collection_name, partition_tag=partition_tag)
        else:
            return self.insert_vectors([vector], [face_id], collection_name=collection_name,
                                       partition_tag=partition_tag)

    def insert_vectors(self, vectors, ids=None, collection_name=None, partition_tag=None):
        """
        insert vectors to search vector engine
        :param vectors: vectors list[list[float]]. example vectors: [[1.2345],[1.2345]]
        :param ids: list of id. If set None, the system will auto generate the ids
        :param collection_name: name of vector collection
        :param partition_tag: partition tag
        :return: list of id
        """

        if collection_name is None:
            raise InternalException("Collection name is empty")

        status, ok = self.milvus.has_collection(collection_name)
        if not ok:
            logger.info("collection not found, create a new collection with collection_name={}".format(collection_name))
            self.create_collection(collection_name)

        if partition_tag is not None:
            self.create_partition(collection_name, partition_tag)

        status, ids = self.milvus.insert(collection_name=collection_name, records=vectors, ids=ids,
                                         partition_tag=partition_tag)

        if status.OK():
            # Flush collection  inserted data to disk.
            self.milvus.flush([collection_name])
            self.get_index_info(collection_name)
            return ids
        else:
            logger.error("Insert failed. {}".format(status))
            self.milvus.delete_entity_by_id(collection_name=collection_name, id_array=ids)
            self.milvus.flush(collection_name_array=[collection_name])
            raise InternalException(status.message)

    def remove_vectors(self, ids, collection_name=None):
        """
        remove vectors from search vector engine
        :param ids: list of id
        :param collection_name: name of vector collection
        :return: boolean
        """
        if collection_name is None:
            raise InternalException("Collection name is empty")

        valid_ids = set()
        for vector_id in ids:
            status, result = self.milvus.get_entity_by_id(collection_name=collection_name, ids=[vector_id])
            if not status.OK():
                raise InternalException(error_code.UNEXPECTED_ERROR)
            if len(result) == 0 or len(result[0]) == 0:
                continue
            valid_ids.add(vector_id)

        if len(valid_ids) == 0:
            # If no id exists in DB, return True
            return True

        valid_ids = list(valid_ids)
        self.milvus.delete_entity_by_id(collection_name=collection_name, id_array=valid_ids)
        self.milvus.flush(collection_name_array=[collection_name])
        self.get_index_info(collection_name)
        status, result = self.milvus.get_entity_by_id(collection_name=collection_name, ids=ids)
        return len(result) == 0 or len(result[0]) == 0

    def remove_vector(self, face_id, collection_name=None):
        """
        remove a vector from search vector engine
        :param face_id: face id
        :param collection_name: name of vector collection
        :return: boolean
        """
        if collection_name is None:
            raise InternalException("Collection name is empty")
        status, result = self.milvus.get_entity_by_id(collection_name=collection_name, ids=[face_id])
        if len(result[0]) == 0:
            raise BadRequestException("ERROR_850", "face_id", face_id)
        self.milvus.delete_entity_by_id(collection_name=collection_name, id_array=[face_id])
        self.milvus.flush(collection_name_array=[collection_name])
        self.get_index_info(collection_name)
        status, result = self.milvus.get_entity_by_id(collection_name=collection_name, ids=[face_id])
        return len(result) == 0 or len(result[0]) == 0

    def is_collection_exists(self, collection_name):
        ok = False
        if not self.milvus:
            self.milvus = self.get_milvus_client()
        else:
            status, ok = self.milvus.has_collection(collection_name)
        return ok
