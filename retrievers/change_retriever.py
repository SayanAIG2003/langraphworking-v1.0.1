# -*- coding: utf-8 -*-

from .config import *

import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
import boto3
import chromadb
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

class ChangeRequestRetriever:
    def __init__(self, embeddings=None, llm=None, chroma_client=None, text_splitter=None):
        """Initialize the retriever with pre-initialized components"""
        try:
            self.embeddings = embeddings
            self.llm = llm
            self.chroma_client = chroma_client

            # Use provided text_splitter or create default
            if text_splitter is None:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
            else:
                self.text_splitter = text_splitter
            
            self.vectorstore = None
            self.retriever = None

            self.initialize_vectorstore()
            self.initialize_self_query_retriever()
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChangeRequestRetriever: {str(e)}")
            
    def _check_existing_database(self) -> bool:
        """Check if database exists and has data"""
        try:
            print("Checking for existing database...")

            collection_name = CHROMA_CONFIG["change_collection_name"] 
           
            if not os.path.exists(CHROMA_CONFIG["persist_directory"]):
                print("Database directory does not exist")
                return False
           
            existing_collections = self.chroma_client.list_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections)
            
            if not collection_exists:
                print(f"Collection '{collection_name}' does not exist")
                return False
            
            collection = self.chroma_client.get_collection(name=collection_name)
            doc_count = collection.count()

            if doc_count == 0 :
                self.chroma_client.delete_collection(name=collection_name)
                print(f"Deleted database with {doc_count} documents")
                return False
            
            print(f"Found existing database with {doc_count} documents")
            return True
               
        except Exception as e:
            print(f"Error checking database: {e}")
            return False
            
    def _convert_date_format(self, date_value, column_name):
        """Convert date to day, month, year integers"""
        try:
            # Handle null/empty values
            if pd.isna(date_value) or str(date_value).strip() in ['', 'nan', 'NaN', 'null']:
                return None, None, None
            
            date_str = str(date_value).strip()
            
            # Define date formats to try (in order of preference)
            date_formats = [
                '%d-%m-%Y',    # 31-07-2025
                '%Y-%m-%d',             # 2025-07-31
                '%d/%m/%Y',             # 31/07/2025
                '%m/%d/%Y',             # 07/31/2025 (US format)
                '%Y/%m/%d',             # 2025/07/31
            ]
            
            # Try each format
            for date_format in date_formats:
                try:
                    parsed_date = pd.to_datetime(date_str, format=date_format, errors='raise')
                    return parsed_date.day, parsed_date.month, parsed_date.year
                except ValueError:
                    continue
            
        except Exception as e:
            print(f"Error converting date '{date_value}' in column '{column_name}': {str(e)}")
            return None, None, None
        
    def preprocess_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date fields to separate day, month, year columns"""
        try:
            print("Processing date fields to separate day/month/year components\n")
            date_columns = ['Opened Date', 'Completed Date']
            
            for col in date_columns:
                if col in df.columns:
                    print(f"Processing date column: {col}")
                    
                    # Create new column names
                    if col == 'Opened Date':
                        day_col, month_col, year_col = 'Opened Day', 'Opened Month', 'Opened Year'
                    else:  # Completed Date
                        day_col, month_col, year_col = 'Completed Day', 'Completed Month', 'Completed Year'
                    
                    # Apply conversion and split into separate columns
                    date_components = df[col].apply(lambda x: self._convert_date_format(x, col))
                    
                    # Create separate columns for day, month, year
                    df[day_col] = date_components.apply(lambda x: x[0] if x[0] is not None else None)
                    df[month_col] = date_components.apply(lambda x: x[1] if x[1] is not None else None)
                    df[year_col] = date_components.apply(lambda x: x[2] if x[2] is not None else None)
                    
                    # Count successful conversions
                    valid_dates = sum(1 for x in date_components if x[0] is not None)
                    print(f"  - Converted {valid_dates} out of {len(df)} dates in '{col}'")
                    
                    # Remove original date column
                    df = df.drop(columns=[col])
                else:
                    print(f"Error: Date column '{col}' not found in data")
        
            return df
            
        except Exception as e:
            raise Exception(f"Error preprocessing dates: {str(e)}")
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
            """Minimal cleaning:
            - run preprocess_dates (creates Opened/Completed Day/Month/Year if present)
            - ensure text columns are strings (empty string for missing)
            - leave numeric columns as-is (no fill with 0)
            """
            try:
                # run existing date preprocessing
                df = self.preprocess_dates(df)
        
                # make sure text columns are strings and missing -> empty string
                string_columns = df.select_dtypes(include=['object']).columns
                df[string_columns] = df[string_columns].fillna('').astype(str)
        
                # leave numeric columns untouched (preserve NaN / pd.NA)
                print(f"Data minimally cleaned. Records: {len(df)}")
                return df
            except Exception as e:
                raise Exception(f"Error cleaning data: {str(e)}")
            
    def load_and_process_data(self) -> List[Document]:
            """Load data and create Documents where selected metadata keys are numeric (int or None).
            - Set numeric columns in `numeric_metadata_keys` (snake_case).
            - Other metadata remain strings (empty string if missing).
            """
            try:
                if not os.path.exists(CHANGE_DATA_FILE_PATH):
                    raise FileNotFoundError(f"Data file not found: {CHANGE_DATA_FILE_PATH}")
        
                df = pd.read_csv(CHANGE_DATA_FILE_PATH)
                print(f"Loaded {len(df)} records from Excel file")
        
                # Run minimal cleaning (includes date splitting if implemented)
                df = self.clean_data(df)
        
                documents = []
        
                # <-- Put any metadata column names you want treated as numeric here (snake_case) -->
                numeric_metadata_keys = {
                    "opened_day", "opened_month", "opened_year",
                    "completed_day", "completed_month", "completed_year", "gear_id"
                }
        
                for idx, row in df.iterrows():
                    try:
                        metadata = {}
                        for col in df.columns:
                            if col.lower() == 'description':
                                continue
        
                            key = col.lower().replace(' ', '_').replace('-', '_')
        
                            if key in numeric_metadata_keys:
                                val = row[col]
                                # preserve missingness as None
                                if pd.isna(val) or str(val).strip() == '':
                                    metadata[key] = None
                                else:
                                    try:
                                        # handle floats that are integral, strings like "2025", numpy ints
                                        metadata[key] = int(float(val))
                                    except Exception:
                                        metadata[key] = None
                            else:
                                metadata[key] = str(row[col]) if pd.notna(row[col]) else ""
        
                        content = (
                            str(row.get('Description', '')) if pd.notna(row.get('Description'))
                            else f"Change ID: {row.get('Change ID', '')}"
                        )
        
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
        
                    except Exception as e:
                        print(f"Error processing row {idx}: {str(e)}")
                        continue
        
                print(f"Created {len(documents)} documents")
                return documents
        
            except Exception as e:
                raise Exception(f"Error loading and processing data: {str(e)}")

    def initialize_vectorstore(self):
            """Initialize or load existing ChromaDB vectorstore"""
            try:
                collection_name = CHROMA_CONFIG["change_collection_name"]
                
                # Check if database already exists using proper collection checking
                if self._check_existing_database():
                    print("Loading existing ChromaDB...")
                    self.vectorstore = Chroma(
                        client=self.chroma_client,  # Use explicit client
                        collection_name=collection_name,
                        embedding_function=self.embeddings
                    )
                    print("Existing database loaded successfully")
                else:
                    print("Creating new ChromaDB...")
                    # Load and process documents
                    documents = self.load_and_process_data()
                    
                    # Split documents if needed
                    split_docs = self.text_splitter.split_documents(documents)
                    print(f"Split into {len(split_docs)} chunks")

                    print("First split metadata:", split_docs[0].metadata)
                    print("opened_year type in split:", type(split_docs[0].metadata.get('opened_year')), split_docs[0].metadata.get('opened_year'))
                    
                    # Create vectorstore with explicit client
                    self.vectorstore = Chroma.from_documents(
                        documents=split_docs,
                        embedding=self.embeddings,
                        client=self.chroma_client,  # Use explicit client
                        collection_name=collection_name
                    )
                    print("New database created successfully")
                    
            except Exception as e:
                raise Exception(f"Error initializing vectorstore: {str(e)}")
            
    def initialize_self_query_retriever(self):
        """Initialize the SelfQueryRetriever"""
        try:
            if self.vectorstore is None:
                raise Exception("Vectorstore not initialized")
            
            # Updated metadata schema with separate date components
            METADATA_FIELD_INFO = [
                {
                    "name": "change_id",
                    "description": "Unique identifier for the change request(e.g. CHG1000943)",
                    "type": "string"
                },
                {
                    "name": "priority",
                    "description": "Priority level of the change (Standard, Emergency, Expedited)",
                    "type": "string"
                },
                # Opened date components
                {
                    "name": "opened_day",
                    "description": "Day of the month when change was opened (1-31)",
                    "type": "integer"
                },
                {
                    "name": "opened_month",
                    "description": "Month when change was opened (1-12)",
                    "type": "integer"
                },
                {
                    "name": "opened_year",
                    "description": "Year when change was opened (e.g., 2024)",
                    "type": "integer"
                },
                # Completed date components  
                {
                    "name": "completed_day",
                    "description": "Day of the month when change was completed (1-31)",
                    "type": "integer"
                },
                {
                    "name": "completed_month",
                    "description": "Month when change was completed (1-12)",
                    "type": "integer"
                },
                {
                    "name": "completed_year",
                    "description": "Year when change was completed (e.g., 2024)",
                    "type": "integer"
                },
                {
                    "name": "status",
                    "description": "Current status of the change (Open, Closed, InProgress, Planned)",
                    "type": "string"
                },
                {
                    "name": "infra_vs_app",
                    "description": "Whether the change is Infra or App",
                    "type": "string"
                },
                {
                    "name": "configuration_item",
                    "description": "The configuration item affected by the change",
                    "type": "string"
                },
                {
                    "name": "assignment_group",
                    "description": "Group assigned to handle the change",
                    "type": "string"
                },
                {
                    "name": "gear_id",
                    "description": "Application number associated with the change",
                    "type": "integer"
                },
                {
                    "name": "application_name",
                    "description": "Name of the application",
                    "type": "string"
                },
                {
                    "name": "segment",
                    "description": "Business segment (EMEA, APAC, NAM, etc.)",
                    "type": "string"
                },
                {
                    "name": "region",
                    "description": "Geographic region",
                    "type": "string"
                },
                {
                    "name": "close_code",
                    "description": "Status of Change(Pending, InProgress, Successful)",
                    "type": "string"
                },
                {
                    "name": "trigger",
                    "description": "What triggered the change (Maintenance, Incident, Project, etc.)",
                    "type": "string"
                },
                {
                    "name": "change_type",
                    "description": "Type of change being made",
                    "type": "string"
                },
                {
                    "name": "related_tickets",
                    "description": "Related incident or request tickets",
                    "type": "string"
                },
                {
                    "name": "avp_required",
                    "description": "Whether AVP approval is required (Y/N)",
                    "type": "string"
                },
                {
                    "name": "cmdb_instance",
                    "description": "CMDB instance identifier",
                    "type": "string"
                },
                {
                    "name": "cmdb_server",
                    "description": "CMDB server identifier",
                    "type": "string"
                },
                {
                    "name": "cmdb_vlan",
                    "description": "CMDB VLAN identifier",
                    "type": "string"
                },
                {
                    "name": "alert",
                    "description": "Alert information",
                    "type": "string"
                }
            ]

            DOCUMENT_CONTENT_DESCRIPTION = "This data source contains the list of all changes implemented or planned to be implemented in the IT landscape of AIG - A global Insurance Company. The fields GEAR_ID and APPLICATION_NAME can be used to identify the changes impacted an application. The change requests implemented or planned could be related to code deployment (new features, fixes), patching and software upgrades, DML (Data updates to any Data residing in a database for an application), DDL or anyother infrastructure changes implemented in the organisation. This change could also be related to any rollback of prior changes implemented. The fields or columns 'Description' and 'Change Type' could be used to understand the type of change"
            
            # Convert metadata field info to AttributeInfo objects
            attribute_info = [
                AttributeInfo(
                    name=field["name"],
                    description=field["description"],
                    type=field["type"]
                ) for field in METADATA_FIELD_INFO
            ]
            
            # Create SelfQueryRetriever
            self.retriever = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                document_contents=DOCUMENT_CONTENT_DESCRIPTION,
                metadata_field_info=attribute_info,
                enable_limit=True,
                verbose=True,
                search_kwargs={"k":55}
            )
            
            print("SelfQueryRetriever initialized successfully")
            
        except Exception as e:
            raise Exception(f"Error initializing SelfQueryRetriever: {str(e)}")
        
    def query(self, user_query: str) -> List[str]:
        """Query the retriever and return change IDs"""
        try:
            if self.retriever is None:
                raise Exception("Retriever not initialized")
            
            # PRINT WHAT QUERY CONSTRUCTOR GENERATES
            try:
                query_constructor = self.retriever.query_constructor
                structured_query = query_constructor.invoke({"query": user_query})
                
                print(f"\n QUERY CONSTRUCTOR OUTPUT:")
                print(f"Query: {structured_query.query}")
                print(f"Filter: {structured_query.filter}")
                print(f"Limit: {structured_query.limit}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Query constructor error: {str(e)}")
            
            # Retrieve documents
            try :
                docs = self.retriever.invoke(user_query)
            except Exception as e:
                raise Exception(f"Error in model: {str(e)}")
            
            # print(docs)
            
            # Extract change IDs
            change_ids = []
            for doc in docs:
                change_id = doc.metadata.get('change_id', '')
                if change_id and change_id not in change_ids:
                    change_ids.append(change_id)
            
            # Save to output file
            output_data = {
                "query": user_query,
                # "formatted_query": formatted_query,
                "total_documents": len(docs),
                "change_ids": change_ids,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            
            print(f"Retrieved {len(docs)} documents")
            print(f"Unique change IDs: {len(change_ids)}")
            
            return change_ids
            
        except Exception as e:
            raise Exception(f"Error during query: {str(e)}")
        
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

        # Extract change IDs
        change_ids = []
        for doc in docs:
            change_id = doc.metadata.get('change_id', '')
            if change_id and change_id not in change_ids:
                change_ids.append(change_id)

        print(len(change_ids),change_ids)

        return docs