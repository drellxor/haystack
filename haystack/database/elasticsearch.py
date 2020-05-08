from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from haystack.database.base import BaseDocumentStore

import logging
logger = logging.getLogger(__name__)


class ElasticsearchDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        host="localhost",
        username="",
        password="",
        index="document",
        search_fields="text",
        text_field="text",
        name_field="name",
        doc_id_field="document_id",
        tag_fields=None,
        embedding_field=None,
        embedding_dim=None,
        custom_mapping=None,
        excluded_meta_data=None,
        scheme="http",
        ca_certs=False,
        verify_certs=True
    ):
        self.client = Elasticsearch(hosts=[host], http_auth=(username, password),
                                    scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs)

        # if no custom_mapping is supplied, use the default mapping
        if not custom_mapping:
            custom_mapping = {
                "mappings": {
                    "properties": {
                        name_field: {"type": "text"},
                        text_field: {"type": "text"},
                        doc_id_field: {"type": "text"},
                    }
                }
            }
            if embedding_field:
                custom_mapping["mappings"]["properties"][embedding_field] = {"type": "dense_vector",
                                                                             "dims": embedding_dim}
        # create an index if not exists
        self.client.indices.create(index=index, ignore=400, body=custom_mapping)
        self.index = index

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        #TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
        # custom fields (e.g. meta data you want to return)
        self.search_fields = search_fields
        self.tag_fields = tag_fields
        self.embedding_field = embedding_field
        self.excluded_meta_data = excluded_meta_data

        text_field_parts = text_field.split('.')
        name_field_parts = name_field.split('.')
        doc_id_field_parts = doc_id_field.split('.')

        if (len(text_field_parts) > 1 or len(doc_id_field_parts) > 1 or len(name_field_parts) > 1) \
                and not (text_field_parts[:-1] == doc_id_field_parts[:-1] == name_field_parts[:-1]):
            raise Exception("text_field, doc_id_field, name_field should have the same nested path")

        self.is_nested = len(text_field_parts) > 1
        self.nested_path = '.'.join(text_field_parts[:-1]) if len(text_field_parts) > 1 else ''
        self.text_field = text_field_parts[-1]
        self.name_field = name_field_parts[-1]
        self.doc_id_field = doc_id_field_parts[-1]



    def get_document_by_id(self, id):
        query = {"filter": {"term": {"_id": id}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {
                "id": result[self.doc_id_field],
                "name": result[self.name_field],
                "text": result[self.text_field],
            }
        else:
            document = None
        return document

    def get_document_by_name(self, name):
        query = {"filter": {"term": {self.name_field: name}}}
        result = self.client.search(index=self.index, body=query)["hits"]["hits"]
        if result:
            document = {
                "id": result[self.doc_id_field],
                "name": result[self.name_field],
                "text": result[self.text_field],
            }
        else:
            document = None
        return document

    def get_document_ids_by_tags(self, tags):
        term_queries = [{"terms": {key: value}} for key, value in tags.items()]
        query = {"query": {"bool": {"must": term_queries}}}
        logger.debug(f"Tag filter query: {query}")
        result = self.client.search(index=self.index, body=query, size=10000)["hits"]["hits"]
        doc_ids = []
        for hit in result:
            doc_ids.append(hit["_id"])
        return doc_ids

    def write_documents(self, documents):
        for d in documents:
            try:
                self.client.index(index=self.index, body=d)
            except Exception as e:
                logger.error(f"Failed to index doc ({e}): {d}")

    def get_document_count(self):
        result = self.client.count()
        count = result["count"]
        return count

    def get_all_documents(self):
        result = scan(self.client, query={"query": {"match_all": {}}}, index=self.index)
        documents = []
        for hit in result:
            documents.append(
                {
                    "id": hit["_source"][self.doc_id_field],
                    "name": hit["_source"][self.name_field],
                    "text": hit["_source"][self.text_field],
                }
            )
        return documents

    def prepare_query(self, search_fields, top_k=10, candidate_doc_ids=None):
        query = {
                "bool": {
                    "should": [{"multi_match": {"query": search_fields, "type": "most_fields", "fields": self.search_fields}}]
                }
            }

        if candidate_doc_ids:
            query["bool"]["filter"] = [{"terms": {"_id": candidate_doc_ids}}]

        if self.is_nested:
            query = {
                "nested": {
                    "path": self.nested_path,
                    "query": query,
                    "inner_hits": {}
                }
            }

        body = {
            "size": top_k,
            "query": query
        }

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        return body

    def extract_paragraphs(self, result):
        paragraphs = []
        meta_data = []
        for hit in result:
            # add the text paragraph
            paragraphs.append(hit["_source"].pop(self.text_field))

            # add & rename some standard fields
            cur_meta = {
                "paragraph_id": hit["_id"],
                "document_id": hit["_source"][self.doc_id_field],
                "document_name": hit["_source"][self.name_field],
                "score": hit["_score"],
                "custom_meta": hit["_source"]
            }
            meta_data.append(cur_meta)
        return paragraphs, meta_data

    def extract_paragraphs_nested(self, result, top_k):
        # Return inner hits instead of full documents
        # Can change to limiting to top_k within cycle if more performance/less memory usage needed
        # Not that necessary for low top_k values
        paragraphs = []
        meta_data = []
        for hit in result:
            for inner_hit in hit["inner_hits"][self.nested_path]["hits"]["hits"]:
                paragraphs.append(inner_hit["_source"][self.text_field])

                cur_meta = {
                    "paragraph_id": hit["_id"],
                    "document_id": inner_hit["_source"][self.doc_id_field],
                    "document_name": inner_hit["_source"][self.name_field],
                    "score": inner_hit["_score"],
                    "custom_meta": inner_hit["_source"]
                }
                meta_data.append(cur_meta)

        # Return only top_k inner hits
        zipped = list(zip(paragraphs, meta_data))
        zipped.sort(key=lambda x: x[1]["score"], reverse=True)
        zipped = zipped[:top_k]
        # Zip returns tuples, library expects lists so...
        paragraphs, meta_data = zip(*zipped)
        return list(paragraphs), list(meta_data)

    def query(self, query, top_k=10, candidate_doc_ids=None):
        # TODO:
        # for now: we keep the current structure of candidate_doc_ids for compatibility with SQL documentstores
        # midterm: get rid of it and do filtering with tags directly in this query
        body = self.prepare_query(query, top_k, candidate_doc_ids)

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=self.index, body=body)["hits"]["hits"]
        if self.is_nested:
            return self.extract_paragraphs_nested(result, top_k)
        else:
            return self.extract_paragraphs(result)

    def query_by_embedding(self, query_emb, top_k=10, candidate_doc_ids=None):
        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in cosine similarity to avoid negative numbers
            body= {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector,doc['question_emb']) + 1.0",
                            "params": {
                                "query_vector": query_emb
                            }
                        }
                    }
                }
            }

            if candidate_doc_ids:
                body["query"]["script_score"]["query"] = {
                    "bool": {
                        "should": [{"match_all": {}}],
                        "filter": [{"terms": {"_id": candidate_doc_ids}}]
                }}

            if self.excluded_meta_data:
                body["_source"] = {"excludes": self.excluded_meta_data}

            logger.debug(f"Retriever query: {body}")
            result = self.client.search(index=self.index, body=body)["hits"]["hits"]
            paragraphs = []
            meta_data = []
            for hit in result:
                # add the text paragraph
                paragraphs.append(hit["_source"].pop(self.text_field))

                # add & rename some standard fields
                cur_meta = {
                        "paragraph_id": hit["_id"],
                        "document_id": hit["_source"].pop(self.doc_id_field),
                        "document_name": hit["_source"].pop(self.name_field),
                        "score": hit["_score"] -1 # -1 because we added +1 in the ES query
                    }
                # add all the rest with original name
                cur_meta.update(hit["_source"])
                meta_data.append(cur_meta)

            return paragraphs, meta_data