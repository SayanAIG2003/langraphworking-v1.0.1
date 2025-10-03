# -*- coding: utf-8 -*-

from .config import *

import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
import boto3
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

class AppCmdbRetriever:
    def __init__(self, embeddings, llm, chroma_client):
        """Initialize the retriever with pre-initialized components"""
        try:
            self.embeddings = embeddings
            self.llm = llm
            self.chroma_client = chroma_client
            
            self.vectorstore = None
            self.retriever = None

            self.initialize_vectorstore()
            self.initialize_retriever()
            
        except Exception as e:
            raise Exception(f"Failed to initialize AppCmdbRetriever: {str(e)}")

    def preprocess_data(self) -> List[Document]:
        """Preprocess the CMDB data and convert to Document objects"""
        try:
            # Read CSV file
            df = pd.read_csv(APP_CMDB_DATA_FILE_PATH)
            print(f"Loaded {len(df)} records from CMDB data file")
            
            # Define major fields that must be present
            major_fields = ['GEAR_ID', 'APPLICATION_NAME', 'SERVER_NAME']
            
            # Replace NaN/empty values with "unknown"
            df = df.fillna("unknown")
            
            documents = []
            skipped_rows = 0
            
            for index, row in df.iterrows():
                # Check if major fields are present and not empty/unknown
                missing_major_fields = []
                for field in major_fields:
                    field_value = str(row.get(field, '')).strip().lower()
                    if field_value in ['', 'nan', 'unknown', 'none']:
                        missing_major_fields.append(field)
                
                if missing_major_fields:
                    skipped_rows += 1
                    continue  # Skip this row as major fields are missing
                
                # Create metadata dictionary with all fields
                metadata = {
                    "server_name": str(row.get('SERVER_NAME', 'unknown')).strip(),
                    "application_name": str(row.get('APPLICATION_NAME', 'unknown')).strip(),
                    "gear_id": str(row.get('GEAR_ID', 'unknown')).strip(),
                    "ci_category": str(row.get('CI_CATEGORY', 'unknown')).strip(),
                    "dv_install_status": str(row.get('DV_INSTALL_STATUS', 'unknown')).strip(),
                    "ip_address": str(row.get('IP_ADDRESS', 'unknown')).strip(),
                    "dv_operational_status": str(row.get('DV_OPERATIONAL_STATUS', 'unknown')).strip(),
                    "dv_support_group": str(row.get('DV_SUPPORT_GROUP', 'unknown')).strip(),
                    "row_id": f"cmdb_row_{index}"
                }
                
                # Create meaningful text content for embedding
                content_parts = []
                
                # Primary information (always include major fields)
                content_parts.append(f"Application: {metadata['application_name']}")
                content_parts.append(f"whose GEAR ID is {metadata['gear_id']} have ")
                content_parts.append(f"Server: {metadata['server_name']}")
                
                # System information
                if metadata['ci_category'] != 'unknown':
                    content_parts.append(f"with Category: {metadata['ci_category']},")
                
                if metadata['ip_address'] != 'unknown':
                    content_parts.append(f"IP Address: {metadata['ip_address']},")
                
                # Status information
                if metadata['dv_install_status'] != 'unknown':
                    content_parts.append(f"Install Status: {metadata['dv_install_status']},")
                    
                if metadata['dv_operational_status'] != 'unknown':
                    content_parts.append(f"Operational Status: {metadata['dv_operational_status']},")
                
                # Support information
                if metadata['dv_support_group'] != 'unknown':
                    content_parts.append(f"Support Group: {metadata['dv_support_group']},")
                
                # Join all parts with separator
                content = " ".join(content_parts)
                
                # Create Document object
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                documents.append(doc)
            
            print(f"Processed {len(documents)} valid App-CMDB records out of {len(df)} total records")
            print(f"Skipped {skipped_rows} records due to missing major fields")
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to preprocess App-CMDB data: {str(e)}")

    def initialize_vectorstore(self):
        """Initialize the vector store, checking for existing collection first"""
        try:
            collection_name = CHROMA_CONFIG["app_cmdb_collection_name"]
            
            # Check if collection exists
            existing_collections = self.chroma_client.list_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections)
            collection = None

            if collection_exists:
                collection = self.chroma_client.get_collection(name=collection_name)

            if collection_exists and collection.count()>0:
                # Load existing vectorstore
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing collection: {collection_name} with total docs: {collection.count()}")
            else:
                print(f"Collection {collection_name} not found. Creating new vectorstore...")

                # Delete any collection still exists
                if collection_exists:
                    try:
                        self.chroma_client.delete_collection(name=collection_name)
                        print(f"Deleted existing collection with 0 doc: {collection_name}")
                    except Exception as e:
                        pass

                # Preprocess data and create new vectorstore
                documents = self.preprocess_data()
                
                if not documents:
                    raise Exception("No valid documents found after preprocessing")
                
                # Create new vectorstore
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    client=self.chroma_client,
                    collection_name=collection_name
                )
                print(f"Created new CMDB vectorstore with {len(documents)} documents")
                
        except Exception as e:
            raise Exception(f"Failed to initialize vectorstore: {str(e)}")

    def initialize_retriever(self):
        """Initialize the self-query retriever with metadata attribute info"""
        try:
            if not self.vectorstore:
                raise Exception("Vectorstore not initialized")
            
            # Define document content description
            document_content_description = "This knowledge base stores the list of all configuration items used by an application. Each application is uniquely identified using GEAR_ID and each configuration item is uniquely identified using SERVER_NAME. The Configuration item could be a server, service or any physical IT Asset."
            
            # Define metadata field attributes for self-query
            metadata_field_info = [
                AttributeInfo(
                    name="server_name",
                    description="The name or hostname of the server",
                    type="string"
                ),
                AttributeInfo(
                    name="application_name", 
                    description="The name of the application",
                    type="string"
                ),
                AttributeInfo(
                    name="gear_id",
                    description="The GEAR ID identifier for the application(e.g., 6412)",
                    type="string"
                ),
                AttributeInfo(
                    name="ci_category",
                    description="The category of the configuration item (e.g., cmdb_ci_apache_web_server, cmdb_ci_linux_server)",
                    type="string"
                ),
                AttributeInfo(
                    name="dv_install_status",
                    description="The installation status of the software/service (e.g., Installed, Retrired, Absent)",
                    type="string"
                ),
                AttributeInfo(
                    name="ip_address",
                    description="The IP address of the server",
                    type="string"
                ),
                AttributeInfo(
                    name="dv_operational_status",
                    description="The operational status of the system (e.g., Operational, Non-Operational, Maintenance)",
                    type="string"
                ),
                AttributeInfo(
                    name="dv_support_group",
                    description="The support group responsible for maintaining the system",
                    type="string"
                ),
                AttributeInfo(
                    name="row_id",
                    description="Unique identifier for the CMDB record",
                    type="string"
                )
            ]
            
            # Create self-query retriever
            self.retriever = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                verbose=True,
                search_kwargs={"k": 50}  # Default number of results
            )
            
            print("Successfully initialized CMDB self-query retriever")
            
        except Exception as e:
            raise Exception(f"Failed to initialize retriever: {str(e)}")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a given query"""
        try:
            if not self.retriever:
                raise Exception("Retriever not initialized")
            
            # Update search kwargs for this query
            self.retriever.search_kwargs = {"k": k}
            
            results = self.retriever.invoke(query)
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if self.vectorstore:
                collection = self.vectorstore._collection
                return {
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
            return {}
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}
        
    def query_filter_construct(self, user_query: str):
        """PRINT WHAT QUERY CONSTRUCTOR GENERATE"""
        try:
            query_constructor = self.retriever.query_constructor
            structured_query = query_constructor.invoke({"query": user_query})
            
            print(f"Filter: {structured_query.filter}")
            print(f"Limit: {structured_query.limit}")
            
        except Exception as e:
            print(f"Query constructor error: {str(e)}")

    def retrieve_docs(self, query: str):

        docs = self.retriever.invoke(query)

        return docs