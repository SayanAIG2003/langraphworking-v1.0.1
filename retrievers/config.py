import os
from typing import Dict, Any

# ChromaDB Configuration
CHROMA_CONFIG = {
    "persist_directory": "./vector_stores",
    "change_collection_name": "change_data",
    "app_cmdb_collection_name": "app_cmdb_data",
    "gear_collection_name": "gear_data",
    "incident_collection_name": "incident_data"
}

# File paths
CHANGE_DATA_FILE_PATH = r"./data_dump/ChangeTicketsForEmbeddings-v6.csv"
APP_CMDB_DATA_FILE_PATH = r"./data_dump/APP-CMDB.csv"
GEAR_DATA_FILE_PATH = r"./data_dump/Gear_data.csv"
INCIDENT_DATA_FILE_PATH = r"./data_dump/Incident_Data.xlsx"