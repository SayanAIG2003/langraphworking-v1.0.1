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
 
class IncidentRetriever:
    def __init__(self, embeddings, llm, chroma_client):
        """Initialize the retriever with pre-initialized components"""
        try:
            self.embeddings = embeddings
            self.llm = llm
            self.chroma_client = chroma_client
 
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
           
            self.vectorstore = None
            self.retriever = None
 
            self.initialize_vectorstore()
            self.initialize_self_query_retriever()
           
        except Exception as e:
            raise Exception(f"Failed to initialize AppCmdbRetriever: {str(e)}")
 
    def _check_existing_database(self) -> bool:
        """Check if database exists and has data"""
        try:
            print("Checking for existing database...")

            collection_name = CHROMA_CONFIG["incident_collection_name"]
           
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
                '%Y-%m-%d %H:%M:%S',    # 2025-07-31 03:59:57
                '%Y-%m-%d',             # 2025-07-31
                '%d-%m-%Y',             # 31-07-2025
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
            date_columns = ['Opened', 'Closed']
           
            for col in date_columns:
                if col in df.columns:
                    print(f"Processing date column: {col}")
                   
                    # Create new column names (avoid conflict with existing 'closed Month')
                    if col == 'Opened':
                        day_col, month_col, year_col = 'Opened_Day', 'Opened_Month', 'Opened_Year'
                    else:  # Closed
                        day_col, month_col, year_col = 'Closed_Day', 'Closed_Month', 'Closed_Year'
                   
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
        - run preprocess_dates (creates Opened/Closed Date Day/Month/Year if present)
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
        
    def create_page_content(self, row):
        """
        Creates human-readable page content combining application name, short description, and description.
        Args: row: pandas Series representing a single incident record 
        Returns: str: Natural language content for semantic search
        """
        content_parts = []
        
        # Short description
        short_desc = row.get('Short description', '')
        if short_desc and str(short_desc).strip():
            content_parts.append(f"Issue is {str(short_desc).strip()}, Details :")
        
        # Detailed description
        description = row.get('Description', '')
        if description and str(description).strip():
            # Clean up the description (remove extra whitespace, normalize line breaks)
            clean_desc = ' '.join(str(description).split())
            content_parts.append(clean_desc)
        
        # Join parts naturally
        if len(content_parts) > 1:
            final_content = ' '.join(content_parts)
        elif content_parts:
            final_content = content_parts[0]
        else:
            # Fallback
            incident_number = row.get('Number', '')
            final_content = f"Incident {incident_number}" if incident_number else "No content available"
        
        return final_content

           
    def load_and_process_data(self) -> List[Document]:
        """Load data and create Documents where selected metadata keys are numeric (int or None).
        - Set numeric columns in `numeric_metadata_keys` (snake_case).
        - Other metadata remain strings (empty string if missing).
        """
        try:
            if not os.path.exists(INCIDENT_DATA_FILE_PATH):
                raise FileNotFoundError(f"Data file not found: {INCIDENT_DATA_FILE_PATH}")
 
            df = pd.read_excel(INCIDENT_DATA_FILE_PATH)
            print(f"Loaded {len(df)} records from Excel file")
 
            # Run minimal cleaning (includes date splitting if implemented)
            df = self.clean_data(df)
 
            documents = []
 
            # Updated numeric metadata keys for incident data (snake_case) - only fields that actually exist
            numeric_metadata_keys = {
                "opened_day", "opened_month", "opened_year",
                "closed_day", "closed_month", "closed_year",
                "gear_id" 
            }
 
            for idx, row in df.iterrows():
                try:
                    metadata = {}
                    for col in df.columns:
                        # Skip fields that shouldn't be metadata (only description and short description)
                        if col.lower() in ['description', 'short description']:
                            continue
 
                        key = col.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
 
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
 
                    content = self.create_page_content(row)

                    if idx==2:
                        print(f"content idx=2 : {content}")
 
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
            # Check if database already exists using proper collection checking
            if self._check_existing_database():
                print("Loading existing ChromaDB...")
                self.vectorstore = Chroma(
                    client=self.chroma_client,  # Use explicit client
                    collection_name=CHROMA_CONFIG["incident_collection_name"],
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
                print("opened_date_year type in split:", type(split_docs[0].metadata.get('opened_date_year')), split_docs[0].metadata.get('opened_date_year'))
               
                # Create vectorstore with explicit client
                self.vectorstore = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    client=self.chroma_client,  # Use explicit client
                    collection_name=CHROMA_CONFIG["incident_collection_name"]
                )
               
                print("New database created successfully")
               
        except Exception as e:
            raise Exception(f"Error initializing vectorstore: {str(e)}")
           
    def initialize_self_query_retriever(self):
        """Initialize the SelfQueryRetriever"""
        try:
            if self.vectorstore is None:
                raise Exception("Vectorstore not initialized")
           
            # Updated metadata schema for incident data - matches actual processed fields
            METADATA_FIELD_INFO = [
                {
                    "name": "number",
                    "description": "Unique incident number identifier (e.g., INC12046965)",
                    "type": "string"
                },
                {
                    "name": "priority",
                    "description": "Priority level of the incident (High, Moderate, Low)",
                    "type": "string"
                },
                {
                    "name": "state",
                    "description": "Current state of the incident (Open/Closed)",
                    "type": "string"
                },
                {
                    "name": "gear_id",
                    "description": "GEAR ID identifier associated with the incident (e.g 6225)",
                    "type": "integer"
                },
                {
                    "name": "application_name",
                    "description": "Name of the application affected by the incident",
                    "type": "string"
                },
                {
                    "name": "ci",
                    "description": "Configuration Item (CI) affected by the incident",
                    "type": "string"
                },
                {
                    "name": "assignment_group",
                    "description": "Support group assigned to handle the incident",
                    "type": "string"
                },
                {
                    "name": "assigned_to",
                    "description": "Individual person assigned to work on the incident",
                    "type": "string"
                },
                {
                    "name": "knowledge_article_used",
                    "description": "Whether a knowledge article was used for resolution (true/false)",
                    "type": "string"
                },
                {
                    "name": "tags",
                    "description": "Tags associated with the incident for categorization",
                    "type": "string"
                },

                {
                    "name": "closed_by",
                    "description": "Person who closed the incident",
                    "type": "string"
                },
                {
                    "name": "impact",
                    "description": "Impact level of the incident (High, Medium, Low)",
                    "type": "string"
                },
                {
                    "name": "urgency",
                    "description": "Urgency level of the incident (High, Medium, Low)",
                    "type": "string"
                },
                # Date components created by preprocess_dates
                {
                    "name": "opened_date_day",
                    "description": "Day of the month when incident was opened (1-31)",
                    "type": "integer"
                },
                {
                    "name": "opened_date_month",  
                    "description": "Month when incident was opened (1-12)",
                    "type": "integer"
                },
                {
                    "name": "opened_date_year",
                    "description": "Year when incident was opened (e.g., 2024, 2025)",
                    "type": "integer"
                },
                {
                    "name": "closed_date_day",
                    "description": "Day of the month when incident was closed (1-31)",
                    "type": "integer"
                },
                {
                    "name": "closed_date_month",
                    "description": "Month when incident was closed (1-12)",
                    "type": "integer"
                },
                {
                    "name": "closed_date_year",
                    "description": "Year when incident was closed (e.g., 2024, 2025)",
                    "type": "integer"
                }
            ]
 
            DOCUMENT_CONTENT_DESCRIPTION = "Incident short description containing details about the problem or issue reported by users"
           
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
        """Query the retriever and return incident numbers"""
        try:
            if self.retriever is None:
                raise Exception("Retriever not initialized")
           
            # PRINT WHAT QUERY CONSTRUCTOR GENERATES
            try:
                query_constructor = self.retriever.query_constructor
                structured_query = query_constructor.invoke({"query": user_query})
               
                print(f"\n🔍 QUERY CONSTRUCTOR OUTPUT:")
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
           
            # Extract incident numbers
            incident_numbers = []
            for doc in docs:
                incident_number = doc.metadata.get('number', '')
                if incident_number and incident_number not in incident_numbers:
                    incident_numbers.append(incident_number)
           
            # Save to output file
            output_data = {
                "query": user_query,
                "total_documents": len(docs),
                "incident_numbers": incident_numbers,
                "timestamp": pd.Timestamp.now().isoformat()
            }
           
            print(f"Retrieved {len(docs)} documents")
            print(f"Unique incident numbers: {len(incident_numbers)}")
           
            return incident_numbers
           
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

        # Extract incident IDs
        incident_ids = []
        for doc in docs:
            incident_id = doc.metadata.get('number', '')
            if incident_id and incident_id not in incident_ids:
                incident_ids.append(incident_id)

        print(len(incident_ids),incident_ids)

        return docs
        