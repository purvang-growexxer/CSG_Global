import base64
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from flask_caching import Cache
from pymongo import MongoClient, UpdateOne
from bson import ObjectId
from datetime import datetime
from datetime import date
# from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import openai
import re
import pandas as pd
# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI, BambooLLM
import os
import numpy as np
import warnings
import time
import snowflake.connector
import configparser
from time import time, sleep
# from fuzzywuzzy import fuzz
import uuid
from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
import pinecone
from transformers import AutoTokenizer, AutoModel
from groq import Groq
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
warnings.filterwarnings('ignore')


# Load configuration from config.ini
config=configparser.ConfigParser()
config.read('config.ini')
chat_gpt_model=config.get('OPENAI','chat_gpt_model')
open_ai_key=config.get('OPENAI','open_ai_key')
gpt_35_model=config.get('OPENAI','gpt_35_model')
gpt_4o_model=config.get('OPENAI','gpt_4o_model')

groq_ai_key=config.get('GROQ','groq_api_key')
groq_model=config.get('GROQ','groq_model')

account=config.get('SNOWFLAKE','account')
username=config.get('SNOWFLAKE','username')
password=config.get('SNOWFLAKE','password')
role=config.get('SNOWFLAKE','role')
schema=config.get('SNOWFLAKE','schema')
warehouse=config.get('SNOWFLAKE','warehouse')
pandasai_key = config.get('PANDASAI','pandasai_key')
mongo_conn_string = config.get('MONGO','mongo_conn_string')
mongo_database = config.get('MONGO','mongo_database')
mongo_collection = config.get('MONGO','mongo_collection')
mongo_database_feedback = config.get('MONGO','mongo_database_feedback')
mongo_collection_feedback = config.get('MONGO','mongo_collection_feedback')
pinecone_api_key = config.get('PINECONE','pinecone_api_key')
index_name = config.get('PINECONE','index_name')
pinecone_host = config.get('PINECONE','pinecone_host')
ddl_pinecone_api_key = config.get('DDL_PINECONE', 'ddl_pinecone_api_key')
ddl_pinecone_index_name = config.get('DDL_PINECONE', 'ddl_pinecone_index_name')

print("chat_gpt_model     : ",chat_gpt_model)
print("open_ai_key        : ",open_ai_key)
print("gpt_35_model       : ",gpt_35_model)
print("gpt_4o_model       : ",gpt_4o_model)
print("groq_ai_key        : ",groq_ai_key)
print("groq_model         : ",groq_model)
print("account            : ",account)
print("username           : ",username)
print("password           : ",password)
print("role               : ",role)
print("schema             : ",schema)
print("warehouse          : ",warehouse)
print("pandasai_key       : ",pandasai_key)
print("mongo_conn_string  : ",mongo_conn_string)
print("mongo_database     : ",mongo_database)
print("mongo_collection   : ",mongo_collection)
print("mongo_database_feedback     : ",mongo_database_feedback)
print("mongo_collection_feedback   : ",mongo_collection_feedback)
print("pinecone_api_key  : ",pinecone_api_key)
print("index_name        : ",index_name)
print("pinecone_host     : ",pinecone_host)

openai.api_key = open_ai_key

app = Flask(__name__)
# Configure caching
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Set Large Language Models (LLMs) for data visualization and chat completion
os.environ['PANDASAI_API_KEY'] = pandasai_key
os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['GROQ_API_KEY'] = groq_ai_key
os.environ['PINECONE_API_KEY'] = pinecone_api_key
# embeddings = OpenAIEmbeddings()

EMBEDDINGS_MODEL_NAME = "thenlper/gte-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

# visualization_llm = OpenAI(api_token=open_ai_key)
database_choice = "CSG"
database = 'CSG'
# llm = openai.OpenAI(api_key=open_ai_key)
g_llm = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
sleep_time = 1

# accesing mongo collection
client = MongoClient(mongo_conn_string)
db = client[mongo_database]
collection = db[mongo_collection]

# accesing mongo collection for feedback
db_feedback = client[mongo_database_feedback]
collection_feedback = db_feedback[mongo_collection_feedback]


vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)
pc=pinecone.Index(api_key=pinecone_api_key,host=pinecone_host)

ddl_pc = pinecone.Pinecone(api_key= ddl_pinecone_api_key)
ddl_index = ddl_pc.Index(name= ddl_pinecone_index_name)

# Creating a logs directory to store application logs if it doesn't already exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# handler = TimedRotatingFileHandler(
#     filename='logs/app.log',
#     when='midnight',  # Rotate log files at midnight
#     interval=1,       # Rotate daily
#     backupCount=7,    # Keep up to 7 log files (including the current one)
# )

handler = RotatingFileHandler(
    filename='logs/app.log',
    maxBytes=1024 * 1024 * 5,  # 5 MB per log file
    backupCount=7,  # Keep up to 7 log files (including the current one)
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler],
)

#-----------------------------------------------------------------------------------------------------------------------
# Executes a SQL query on a Snowflake database and returns the result as a DataFrame
def run_query(database,query):
    try:
        conn = snowflake.connector.connect(
                    user=username,
                    password=password,
                    account=account,
                    warehouse=warehouse,
                    database=database,
                    role=role
                )
        df = pd.read_sql(query, conn)
        return "successfully executed",df
    except snowflake.connector.Error as e:
        print(f"Snowflake Error: {e}")
        return "error while executing",pd.DataFrame()
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return "error while executing",pd.DataFrame()

# Parses a SQL CREATE TABLE statement and extracts table schema, primary keys, foreign keys and column data types
def parse_sql_create_table(input_sql, table_name, db):
    # Initialize dictionary structure
    output = {
        "table_name": table_name,
        "schema": f'{db}.PUBLIC',
        "primary_key": []
    }
    datatype_mapping = {
        "NUMBER": "number_datatype_columns",
        "NUMERIC": "number_datatype_columns",
        "DECIMAL": "decimal_datatype_columns",
        "INT": "integer_datatype_columns",
        "INTEGER": "integer_datatype_columns",
        "BIGINT": "bigint_datatype_columns",
        "SMALLINT": "smallint_datatype_columns",
        "TINYINT": "tinyint_datatype_columns",
        "DOUBLE": "double_datatype_columns",
        "REAL": "real_datatype_columns",
        "FLOAT": "float_datatype_columns",
        "VARCHAR": "varchar_datatype_columns",
        "CHAR": "char_datatype_columns",
        "TEXT": "text_datatype_columns",
        "CLOB": "clob_datatype_columns",
        "DATE": "date_datatype_columns",
        "TIME": "time_datatype_columns",
        "TIMESTAMP": "timestamp_datatype_columns",
        "TIMESTAMP_NTZ": "timestamp_ntz_datatype_columns",
        "TIMESTAMP_LTZ": "timestamp_ltz_datatype_columns",
        "TIMESTAMP_TZ": "timestamp_tz_datatype_columns",
        "INTERVAL": "interval_datatype_columns",
        "BOOLEAN": "boolean_datatype_columns",
        "BINARY": "binary_datatype_columns",
        "VARBINARY": "varbinary_datatype_columns",
        "BLOB": "blob_datatype_columns",
        "XML": "xml_datatype_columns",
        "JSON": "json_datatype_columns",
        "GEOGRAPHY": "geography_datatype_columns"
    }

    # Initialize lists in the output based on the datatype mapping
    for key in datatype_mapping.values():
        output[key] = []

    # Extract table name and schema
    table_match = re.search(r"create or replace TABLE (\w+\.\w+)\.(\w+)", input_sql, re.IGNORECASE)
    if table_match:
        output["schema"], output["table_name"] = table_match.groups()

    # Process each column line separately
    columns_part = input_sql.split("(", 1)[1].rsplit(")", 1)[0]
    column_lines = [line.strip() for line in columns_part.split(",") if line.strip()]

    # Regular expressions for detecting column types and constraints
    column_regex = r"^(\w+) (\w+)"
    pk_regex = r"primary key \(([\w, ]+)\)"
    fk_regex = re.compile(r'constraint (\w+) foreign key \((\w+)\) references ([\w\.]+)\((\w+)\)')

    foreign_keys = []
    # Process each line within the column definition
    for line in column_lines:
        column_match = re.match(column_regex, line, re.IGNORECASE)
        if column_match:
            column_name, data_type = column_match.groups()
            # Check if the data type is found in our mapping dictionary
            found = False
            for dt_key, dt_list in datatype_mapping.items():
                if dt_key in data_type.upper():
                    output[dt_list].append(column_name)
                    found = True

        pk_match = re.search(pk_regex, line, re.IGNORECASE)
        if pk_match:
            # Handle composite primary keys
            keys = [key.strip() for key in pk_match.group(1).split(',')]
            output["primary_key"].extend(keys)

    # Handle foreign keys outside the loop to avoid repeated parsing
    fk_match = fk_regex.findall(input_sql)
    for idx, match in enumerate(fk_match, start=1):
        _, column, reference_table, reference_column = match
        fk_key = f"foreign_key_{idx}"
        output[fk_key] = [column]
        output[f"{fk_key}_reference"] = {'table': reference_table, 'column': reference_column}

    output = {k: v for k, v in output.items() if v is not None and v != [] and v != {}}
    return output

# Adds quotes to the string and list values in a dictionary, excluding the key 'schema'
def add_quotes_to_dict_values(input_dict):
    quoted_dict = {}
    for key, value in input_dict.items():
        if key != 'schema':
            if isinstance(value, str):
                quoted_dict[key] = f'"{value}"'
            elif isinstance(value, list):
                quoted_dict[key] = [f'"{item}"' for item in value]
            else:
                quoted_dict[key] = value
        else:
            quoted_dict[key] = value
    return quoted_dict

# Retrieves sample data from each table in the specified database and summarizes unique values
def get_sample_data(database,df):
    xxx = ""
    all_ = []
    all_table_sample_data = {}

    for table_name in df['TABLE_NAME']:
        query = f"SELECT * FROM {database}.PUBLIC.\"{table_name}\" LIMIT 50"
        execution_status,df_table = run_query(database,query)
        if len(df_table)!=0:
            xxx = xxx  + 'Table_name = ' + f"\"{table_name}\"" + f' :: Schema: {database}.PUBLIC \n'
            for idx,column in enumerate(df_table.columns):
                unique_values = df_table[column].unique()
                if unique_values.dtype == np.dtype('int64')  or unique_values.dtype == np.dtype('float64'):
                    pass
                else:
                    if len(unique_values) <15:
                        xxx = xxx + "{}. {} :: Unique_values_count :: {}".format(idx,column, len(unique_values)) + f" :: {unique_values[:15]}" + '\n'
                    else:
                        temp = [str(e) for e in (df_table[column][:])]
                        temp = list(set(temp))
                        temp = temp[:5]
                        xxx = xxx + "{}. {} :: Unique_values_count :: {}".format(idx,column, len(unique_values)) + f" :: {temp}"[:-1] + ", ......]" + '\n'
            xxx = xxx + '*'*10
            all_table_sample_data[f"\"{table_name}\""] = xxx
            all_.append(xxx)
            xxx=''
    return all_table_sample_data,all_

# Retrieves and parses the schema for each table in the specified database and formats it into a list and string
def get_schema(database,df):
    all_table_schema_list = []
    all_table_schema_string = """"""
    for table_name in df['TABLE_NAME']:
        query = f"select GET_DDL ('TABLE', '\"{table_name}\"');"
        execution_status,df_table = run_query(database,query)
        input_sql = df_table.values[0][0]
        input_sql = re.sub(r'"', '', input_sql)
        output = parse_sql_create_table(input_sql, table_name=table_name, db=database)
        output = add_quotes_to_dict_values(output)
        all_table_schema_list.append(output)
        all_table_schema_string = all_table_schema_string + str(output) + ', \n\n'
    return all_table_schema_list,all_table_schema_string

# Retrieves table names from the database and gathers schema and sample data for each table
def get_database_info_call(database):
    #get all the table names present in the database
    query = f"SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = '{schema}'"
    execution_status,df = run_query(database,query)

    if df.empty:
        print(f"no data present for database : {database}")
        all_table_schema_list = []
        all_table_schema_string = ''
        all_table_sample_data = {}
        all_ = []
    else:
        #get sample data of each table
        print(f"getting sample data for database {database}")
        all_table_sample_data,all_ = get_sample_data(database,df)
        #get schema of each table in specific format
        print(f"getting schema for each table in database {database}")
        all_table_schema_list,all_table_schema_string = get_schema(database,df)
        print(f"schema and sample data is retrieved for database {database}")

    print("--------------------")
    # print(all_table_sample_data)
    # print(all_table_schema_list)
    # print(all_table_schema_string)
    print("--------------------")

    return all_table_schema_list, all_table_schema_string, all_table_sample_data, df, all_
#-----------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------
# Generates a chat completion response using a specified OpenAI language model
# model will be selected from frontend
def chat_completion(prompt,model):
    if model == 'Llama3':
        completion = g_llm.chat.completions.create(
            messages=[{"role": "system","content": prompt}],
            model=groq_model
        )
        return completion.choices[0].message.content
    elif model == 'ChatGPT':
        completion = openai.ChatCompletion.create(
            model=gpt_4o_model,
            messages=[{"role": "system", "content": prompt}],
            temperature=1
        )
        return completion.choices[0].message.content
    else:
        return 'Error in chat_completion function'

def vectorize_text(text: str) -> np.ndarray:
    """
    Vectorizes the input text using a pre-trained model.

    Parameters:
    text (str): The input text to be vectorized.

    Returns:
    np.ndarray: The vector representation of the input text.
    """
    if not hasattr(vectorize_text, "tokenizer"):
        model_name = 'Alibaba-NLP/gte-base-en-v1.5'
        vectorize_text.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        vectorize_text.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    inputs = vectorize_text.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = vectorize_text.model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def get_related_tables(index, query, filter_conditions=None, top_k=20):
    """
    Queries Pinecone with the provided query and filter conditions & returns related tables.

    Parameters:
    - index: The Pinecone index to query.
    - query: The text query to be vectorized and used for querying Pinecone.
    - filter_conditions: A dictionary of conditions to filter the results. Default is None.
    - top_k: The number of top results to return. Default is 10.

    Returns:
    - A list of unique table names extracted from the metadata of the query results.
    """

    query_embedding = vectorize_text(query)

    # Prepare the query parameters
    query_params = {
        "vector": query_embedding.tolist(),
        "top_k": top_k,
        "include_values": True,
        "include_metadata": True
    }

    # Add filter conditions if provided
    if filter_conditions:
        query_params["filter"] = filter_conditions

    # Query Pinecone
    query_results = index.query(**query_params)

    # Extract unique table names from the metadata
    unique_table_names = set()
    for result in query_results['matches']:
        metadata = result.get('metadata', {})
        table_name = metadata.get('table')
        if table_name:
            unique_table_names.add(table_name)

    # Return the list of unique table names
    return list(unique_table_names)

# generates the sql queries different variation using database's schema,sample data by multiple OpenAI calls
def open_ai_call(question,model_choice,all_table_schema_string, df, all_, all_table_schema_list):
    print(f"llm call for question '{question}' using model : {model_choice}.. ")

    # if model_choice == "TEST_LLM_NIKUL_DATA":
    #     schema_prompt = all_table_schema_string
    #     schema_prompt = """roleMappings: {
    #         1: 'STUDENT',
    #         2: 'TEACHER',
    #         3: 'ADMIN'
    #     }""" + schema_prompt
    # else:
    schema_prompt = all_table_schema_string
    print('schema_prompt :: ', schema_prompt)
    # print("schema_prompt : ",schema_prompt)

    database_hint = 'Snowflake'

    # Second call to the Language Model (LLM) to generate a query using a question, DDL of ROI tables and sample data of ROI tables
    st = time()
    # print(all_)
    sample_data_1 = all_
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(sample_data_1)

    # filter_conditions = {
    # "database": {"$eq": model_choice}
    # }

    # roi_tables = get_related_tables( ddl_index, query = question, filter_conditions=filter_conditions, top_k=20)
    roi_tables = ['GLOBALINVOICES']
    # print("ROI TABLES:")
    # print(roi_tables)
    final_sample_data = []
    for table in roi_tables:
        for sd in sample_data_1:
            if table in sd:
                final_sample_data.append(sd)

    print("\nafter final sample data append for loop\n")

    final_ddl_data = """create or replace TABLE CSG.PUBLIC.GLOBALINVOICES (
	BUSINESS_UNIT_NAME VARCHAR(16777216),
	SUPPLIER_NUMBER NUMBER(38,0),
	SUPPLIER_NAME VARCHAR(16777216),
	PO_NUMBER VARCHAR(16777216),
	PO_START_DATE DATE,
	PO_END_DATE DATE,
	PO_CURRENCY VARCHAR(16777216),
	PO_ORIGINAL_AMOUNT FLOAT,
	OVERALL_PO_CONSUMED_AMOUNT FLOAT,
	OVERALL_PO_AMOUNT_REMAINING FLOAT,
	PO_UNITPRICE FLOAT,
	PO_REQUESTER VARCHAR(16777216),
	PO_REQUESTER_EMAIL VARCHAR(16777216),
	PO_REQ_END_DATE DATE,
	PO_REQUESTER_SUPERVISOR VARCHAR(16777216),
	MATCHED_PO_AMOUNT FLOAT,
	MATCHED_PO_QUANTITY FLOAT,
	INVOICE_NUMBER VARCHAR(16777216),
	INVOICE_AMOUNT FLOAT,
	PAYMENT_STATUS VARCHAR(16777216),
	HOLD_STATUS VARCHAR(16777216),
	INVOICE_VALIDATION_STATUS VARCHAR(16777216),
	INVOICE_APPROVAL_STATUS VARCHAR(16777216),
	CURRENT_APPROVER VARCHAR(16777216),
	CURRENT_APPROVER_END_DATE DATE
    );"""
    # for table in roi_tables:
    #     for sd in all_table_schema_list:
    #         if sd['table_name'] == table:
    #             final_ddl_data.append(sd)

    # if len(final_ddl_data) == 0:
    #     final_ddl_data = all_table_schema_list
    if len(final_sample_data) == 0:
        final_sample_data = sample_data_1


    print("\nafter final ddl data and before calling prompt_1\n")

    # Third call to the Language Model (LLM) to get the elaboration of a question using question, DDL of ROI tables and sample data of ROI tables
    st = time()
    prompt_1 = f"""
    Using the provided DDL and sample data, Your job is to re-write an elaborated, detailed verison of the given Question.

    # Question
    {question}

    # DDL
    {final_ddl_data}

    # Database_Sample_Data
    {final_sample_data}

    Instructions:
    - Use the column names and data types from the DDL to add specificity and context to the question.
    - Ensure the core intent of the original question is preserved while making it more comprehensive.
    - Present only the question as the output. Exclude any explanations, annotations, or step-by-step descriptions.

    Available tables : {', '.join(roi_tables)}
    """

    response_1 = chat_completion(prompt_1,model_choice)
    print("response_1   : ",response_1)
    et = time()
    llm_response_time_1 = round((et-st), ndigits=4)
    print("llm_response_time_1 : ",llm_response_time_1)
    sleep(sleep_time)

    print("\nafter prompt_1 and before prompt_2\n")

    prompt_2 = f"""
    You are an expert in SQL query generation. Your task is to create a precise and efficient {database_hint} SQL query based on the provided Question, Database Sample Data and DDL.

    # Question
    {response_1}

    # Database Sample Data
    {final_sample_data}

    # DDL
    {final_ddl_data}

    Instructions for generating SQL Query:
    - Ensure to use the DDL as the main source of truth to understand table structures, column names, data types, and relationships.
    - Generate a clear, accurate, and non-ambiguous {database_hint} SQL query that addresses the given Question.
    - Avoid unnecessary complexity and ensure the query directly answers the question.
    - Provide only the final SQL query. Do not include any explanations, comments, or additional information.

    Available tables : {', '.join(roi_tables)}
    """

    response_2 = chat_completion(prompt_2,model_choice)
    print("response_2   : ",response_2)
    et = time()
    llm_response_time_2 = round((et-st), ndigits=4)
    print("llm_response_time_2 : ",llm_response_time_2)
    sleep(sleep_time)

    return response_1, response_2, final_sample_data, final_ddl_data, ', '.join(roi_tables)


#-----------------------------------------------------------------------------------------------------------------------

import re
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def find_sql_queries(text):
#     """
#     Splits the input SQL string into individual SQL statements.

#     Parameters:
#     text (str): A string containing one or more SQL statements.

#     Returns:
#     list: A list of SQL statements, each ending with a semicolon.

#     Raises:
#     ValueError: If the input is not a string or is empty.
#     """
#     # Check if the input is a string
#     if not isinstance(text, str):
#         logging.error("Input is not a string.")
#         raise ValueError("Input must be a string containing SQL statements.")

#     # Check if the input string is empty
#     if not text.strip():
#         logging.error("Input string is empty.")
#         raise ValueError("Input string cannot be empty.")

#     try:
#         # Define the SQL pattern
#         sql_pattern = re.findall(r'((SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)[\s\S]+?;)', text, re.IGNORECASE)

#         # Extract full SQL statements from the tuples returned by re.findall
#         sql_statements = [match[0] for match in sql_pattern]

#         # Logging the number of statements found
#         logging.info(f"Found {len(sql_statements)} SQL statements.")

#         return sql_statements

#     except re.error as regex_error:
#         logging.error(f"Regex error occurred: {regex_error}")
#         raise ValueError("An error occurred while processing the SQL statements.") from regex_error

#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         raise RuntimeError("An unexpected error occurred while processing the SQL statements.") from e

def find_sql_queries(text):
    """
    Splits the input SQL string into individual SQL statements.

    Parameters:
    text (str): A string containing one or more SQL statements.

    Returns:
    list: A list of SQL statements, each ending with a semicolon.

    Raises:
    ValueError: If the input is not a string or is empty.
    """
    # Check if the input is a string
    if not isinstance(text, str):
        logging.error("Input is not a string.")
        raise ValueError("Input must be a string containing SQL statements.")

    # Check if the input string is empty
    if not text.strip():
        logging.error("Input string is empty.")
        raise ValueError("Input string cannot be empty.")

    try:
        queries = text.strip().strip("`").strip()
        print("printing queries in find_sql_queries :: ", queries)
        # Split the input text into lines and remove any unnecessary whitespace lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Re-join lines with a single space, maintaining the logical structure of the SQL statement
        normalized_text = " ".join(lines)

        # Check if the string ends with a semicolon, if not add it
        if not normalized_text.strip().endswith(";"):
            normalized_text = normalized_text.strip() + ";"

        print("printing queries again in find_sql_queries :: ", queries)

        # Define the SQL pattern
        sql_pattern = re.findall(r'((WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)[\s\S]+?;)', normalized_text, re.IGNORECASE)

        # Extract full SQL statements from the tuples returned by re.findall
        sql_statements = [match[0] for match in sql_pattern]

        # Logging the number of statements found
        logging.info(f"Found {len(sql_statements)} SQL statements.")

        return sql_statements

    except re.error as regex_error:
        logging.error(f"Regex error occurred: {regex_error}")
        raise ValueError("An error occurred while processing the SQL statements.") from regex_error

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise RuntimeError("An unexpected error occurred while processing the SQL statements.") from e


#To find the sql queries from the given text
# def find_sql_queries(text):
#     queries = re.findall(r'```sql(.*?)```', text, re.DOTALL)
#     queries = [query.strip() for query in queries]
#     if not queries:
#         return text

#     list_of_queries = []
#     for query in queries:
#         list_of_queries.append(query)
#     return list_of_queries

# def find_sql_queries(text):
#     # Find SQL queries enclosed in triple backticks
#     queries = re.findall(r'```sql(.*?)```', text, re.DOTALL)
#     queries = [query.strip() for query in queries]

#     # If no queries are found within backticks, use a fallback strategy
#     if not queries:
#         # Try to extract SQL-like statements using a basic heuristic (like detecting common SQL keywords)
#         sql_pattern = re.findall(r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)[\s\S]+?;', text, re.IGNORECASE)
#         sql_pattern = [query.strip() for query in sql_pattern]

#         # If no SQL-like patterns are found, return the original text
#         if not sql_pattern:
#             return text

#         return sql_pattern

#     # If backtick-enclosed queries are found, return them
#     return queries

#get the list of successfully running queries and its output
def get_running_queries(choice,queries):
    compare_queries = []
    df_list = []
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(queries)
    for query in queries:
        try:
            execution_status,df_result = run_query(choice,query)
            if execution_status=="error while executing":
                print("--")
                pass
            else:
                print("++")
                # dump here a list of dfs
                compare_queries.append(query)
                df_list.append(df_result)
        except:
            # st.write("Please re-run the query")
            print("Please re-run the query")
    return compare_queries,df_list

# Function to compare DataFrames
def find_duplicates_and_unique(df_list):
    duplicates = []
    unique_dfs = []
    unique_dfs.append(df_list[0])

    for i in range(1, len(df_list)):
        is_duplicate = False
        for j in range(len(unique_dfs)):
            if df_list[i].equals(unique_dfs[j]):
                duplicates.append((i, j))
                is_duplicate = True
                break
        if not is_duplicate:
            unique_dfs.append(df_list[i])

    return duplicates, unique_dfs

# Function to remove redundant queries
def remove_elements_at_indices(compare_queries, a):
    # Sort the indices in reverse order to avoid indexing issues
    indices_to_remove = sorted(a, reverse=True)
    # Remove elements at the specified indices
    for index in indices_to_remove:
        if 0 <= index < len(compare_queries):
            compare_queries.pop(index)
    return compare_queries

#ranking of the remaining queries in correctness order
def rank_sql_queries(question, queries, ddl, sample_data,gpt_35_model):

    prompt = f"""
    # Question:
    {question}

    # SQL Queries:
    {queries}

    # DDL:
    {ddl}

    # Sample Data:
    {sample_data}

    # Evaluation Criteria: accuracy

    Please evaluate the above SQL queries based on the given question, sample data, and DDL. Rank them according to accuracy. Only return the most accurate SQL query from those 5 queries.

    # DO NOT CHANGE THE ORIGINAL QUERIES.

    # Return only the best rated SQL query. No extra information.

    # Only SQL Query
    """

    response = g_llm.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": "You are a database expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return question, response.choices[0].message.content

# Function to convert columns to their dedicated data types
def convert_to_dedicated_dtypes(df):
    """
    Store a DataFrame to CSV, read it back, and delete the CSV file.

    Parameters:
    df (pd.DataFrame): DataFrame to be stored in the CSV file.

    Returns:
    pd.DataFrame: DataFrame read from the CSV file with dedicated datatypes or the input DataFrame in case of an error.
    """
    try:
        # Get the current working directory
        current_dir = os.getcwd()
        # Define the file path
        file_path = os.path.join(current_dir, 'df_for_visualization.csv')

        # Store DataFrame to CSV
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")

        # Read the CSV back into a DataFrame with their dedicated datatypes
        df_read = pd.read_csv(file_path)
        print("DataFrame read from CSV:")
        print(df_read)

        # Delete the CSV file
        os.remove(file_path)
        print(f"CSV file {file_path} deleted")

        return df_read

    except Exception as e:
        print(f"An error occurred in saving or reading the dataframe with actual datatypes: {e}")
        return df

#-----------------------------------------------------------------------------------------------------------------------

model_mappings = {
    'Llama3':'Llama3',
    'ChatGPT':'ChatGPT'
}

database_mappings = {
        'CSG_DATA':'CSG'
# 'TEST_LLM_SIMPLYMUSIC_DATA': 'CSG',
# 'TEST_LLM_PSI_DATA': 'CSG',
# 'TEST_LLM_MATANGI_DATA': 'CSG'
}

# Dictionary to store cached database info
cached_data = {
        'CSG_DATA': None
# 'TEST_LLM_SIMPLYMUSIC_DATA': None,
# 'TEST_LLM_PSI_DATA': None,
# 'TEST_LLM_MATANGI_DATA': None,
}

# upload embeddings to pinecone index
def upload_embeddings(question,query,database,question_key,pinecone_api_key):
    print(question,query)
    vectors_info=pinecone.Index.query(self=pc,
                    vector=list(np.zeros(1536)),
                    filter={"question_key": question_key},
                    top_k=10000,
                    include_metadata=False,api_key=pinecone_api_key)
    vectors_ids = [entry['id'] for entry in vectors_info['matches']]
    print("len of vectors_ids : ",len(vectors_ids))
    if vectors_ids:
        print(f"vector for key {question_key} is present.")
    else:
        if query is None:
            query = "None"
        # vectorstore.add_texts(texts=["give me employee name who has second highest salary"],metadatas=[{"query":"select * from invoice","database":"matangi_database"}])
        vectorstore.add_texts(texts=[question],metadatas=[{"query":query,"database":database,"question_key":question_key}])
        print(f"Uploaded vectors for key {question_key}.")

# delete embeddings from pinecone index
def delete_vectors_by_ids(question_key,pinecone_api_key):
    vectors_info=pinecone.Index.query(self=pc,
                     vector=list(np.zeros(1536)),
                     filter={"question_key": question_key},
                     top_k=10000,
                     include_metadata=False,api_key=pinecone_api_key)
    vectors_ids = [entry['id'] for entry in vectors_info['matches']]
    print("len of vectors_ids : ",len(vectors_ids))
    if vectors_ids:
        pinecone.index.Index.delete(self=pc,ids=vectors_ids)
        print(f"deleted {len(vectors_ids)} vectors")
    else:
        print(f"vectors with question_key {question_key} are not present.")

#counts no. of 'yes' and 'no' in string
def count_yes_no(s):
    words = s.strip().split()
    yes_count = sum(1 for word in words if word.lower() == 'yes')
    no_count = sum(1 for word in words if word.lower() == 'no')
    return yes_count, no_count

def generate_natural_language_response(user_question, df_result):
    # Format the prompt with both the user question and the query result
    prompt = (
        # f"You are a database expert. A user has asked the following question: '{user_question}'. "
        # f"The query has returned the following result: {query_result}. "
        # f"Please provide a clear and concise explanation of the result in plain English, making it easy for a user to understand."

        f"You are an expert data analyst. Here is the context:"
        f"Question: {user_question}"
        f"SQL Result: {df_result}"
    
        f"Please generate a concise and user-friendly natural language summary of the SQL result based on the question."
    )
 
    try:
        response = g_llm.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in databases."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Return the natural language response from the model's output
        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error generating natural language response: {e}")
        return "Sorry, I couldn't generate a response."
 

@app.route('/feedback', methods=['POST'])
@cross_origin(supports_credentials=True)
def submit_query():
    try:
        print("A POST request was received at the /feedback endpoint.")
        logging.info("A POST request was received at the /feedback endpoint.")
        data = request.json
        key = data.get('key')
        question_key = data.get('question_key')
        database = data.get('database')
        question = data.get('question')
        query = data.get('query')
        feedback_value = data.get('feedback')

        # Validate the input
        if any(field is None for field in [key, question_key, database, question, feedback_value]):
            return jsonify({"error": "All fields are required"}), 400

        if feedback_value==1:
            # upload_embeddings(question,query,database,question_key)
            upload_embeddings(question,query,database,question_key,pinecone_api_key)
            # Create a document to store in MongoDB
            document = {
                "upload_emebedding_status": 1,
                "key": key,
                "question_key": question_key,
                "database": database,
                "question": question,
                "query": query,
                "feedback_value": feedback_value
            }
        else:
            delete_vectors_by_ids(question_key,pinecone_api_key)
            document = {
                "upload_emebedding_status": 0,
                "key": key,
                "question_key": question_key,
                "database": database,
                "question": question,
                "query": query,
                "feedback_value": feedback_value
            }

        # Check if a document with the provided key and question_key exists
        existing_document = collection_feedback.find_one({"key": key, "question_key": question_key})

        if existing_document:
            # Update the existing document
            collection_feedback.update_one(
                {"key": key, "question_key": question_key},
                {"$set": document}
            )
            return jsonify({"status": 1, "data": "Data updated successfully"})
        else:
            # Insert the document into the collection
            collection_feedback.insert_one(document)
            return jsonify({"status": 1, "data": "Data received and stored successfully"})
    except Exception as e:
        print(f"An error occurred while processing a POST request at the /feedback endpoint  : {e}")
        logging.error(f"An error occurred while processing a POST request at the /feedback endpoint  : {e}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}), 400


@app.route('/get_feeback_status', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_document():
    try:
        print("A GET request was received at the /get_feeback_status endpoint.")
        logging.info("A GET request was received at the /get_feeback_status endpoint.")
        # Get query parameters
        key = request.args.get('key')
        question_key = request.args.get('question_key')
        print("key : ",key)
        print("question_key : ",question_key)

        if not key or not question_key:
            return jsonify({'error': 'Both key and question_key are required'}), 400

        # Query the MongoDB collection
        document = collection_feedback.find_one({'key': key, 'question_key': question_key})

        if document:
            # Convert BSON ObjectId to string for JSON serialization
            document['_id'] = str(document['_id'])
            return jsonify({"status": 1, "data": document})
        else:
            return jsonify({"status": 1, "data": []})
    except Exception as e:
        print(f"An error occurred while processing a GET request at the /get_feeback_status endpoint  : {e}")
        logging.error(f"An error occurred while processing a GET request at the /get_feeback_status endpoint for key {key}: {str(e)}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}),400


@app.route('/feedback/delete', methods=['GET'])
@cross_origin(supports_credentials=True)
def delete_document():
    try:
        print("A GET request was received at the /feedback/delete endpoint.")
        logging.info("A GET request was received at the /feedback/delete endpoint.")
        # Get query parameters
        key = request.args.get('key')
        question_key = request.args.get('question_key')

        if not key or not question_key:
            return jsonify({'error': 'Both key and question_key are required'}), 400

        delete_vectors_by_ids(question_key,pinecone_api_key)
        # Query the MongoDB collection and delete the document
        result = collection_feedback.delete_many({'key': key, 'question_key': question_key})
        #delete embeddings from pinecone function

        if result.deleted_count > 0:
            return jsonify({"status": 1, "message": f'{result.deleted_count} document(s) deleted successfully'})
        else:
            return jsonify({"status": 0, "message": f"No document found with the provided key {key} and question_key {question_key}."})
    except Exception as e:
        print(f"An error occurred while processing a GET request at the /feedback/delete endpoint  : {e}")
        logging.error(f"An error occurred while processing a GET request at the /feedback/delete endpoint for key {key}: {str(e)}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}),400


# Encodes datetime,date and ObjectId objects to their string representations for JSON serialization
def default_encoder(obj):
    if isinstance(obj, (datetime, date, ObjectId)):
        return str(obj)
    raise TypeError("Object not serializable")

def cache_results():
    for cache_key, database in database_mappings.items():
        try:
            schema_list, schema_string, sample_data, df, all_data = get_database_info_call(database)
            cached_data[cache_key] = {
                'schema_list': schema_list,
                'schema_string': schema_string,
                'sample_data': sample_data,
                'df': df,
                'all': all_data,
            }
        except Exception as e:
            print("error while getting the schema and sample data of database")


print("cached_data before: ",cached_data)
# Call the cache_results function directly
cache_results()
print("cached_data after: ",cached_data)


@app.route('/options', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_options():
    try:
        print("A GET request was received at the /options endpoint.")
        logging.info("A GET request was received at the /options endpoint.")
        options = list(model_mappings.keys())
        print("Successfully retrieved options")
        logging.info("Successfully retrieved options")
        return jsonify({"status": 1, "data": options})
    except Exception as e:
        print(f"An error occurred while processing a GET request at the /options endpoint  : {e}")
        logging.error(f"An error occurred while processing a GET request at the /options endpoint  : {e}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}), 400

def save_chat_to_database(key, chat_data):
    # Deserialize chat_data if it's already JSON serialized
    if isinstance(chat_data, str):
        chat_data = json.loads(chat_data)

    existing_chat = collection.find_one({'key': key})
    if existing_chat:
        chat_history = existing_chat.get('chat_history', [])
        # Include the key with the chat_data
        chat_data_with_key = {'key': key, **chat_data}
        chat_history.append(chat_data_with_key)
        collection.update_one({'_id': existing_chat['_id']}, {'$set': {'chat_history': chat_history}})
        logging.info(f"Chat history appended for key {key}")
    else:
        chat_document = {
            'key': key,
            **chat_data,
            'chat_history': []
        }
        print("chat_document : ",chat_document)
        collection.insert_one(chat_document)
        logging.info(f"No chat history found for key {key}. New chat document inserted.")

@app.route('/chat', methods=['POST'])
@cross_origin(supports_credentials=True)
def process_query():
    try:
        print("A POST request was received at the /chat endpoint.")
        logging.info("A POST request was received at the /chat endpoint.")
        data = request.get_json()
        key = data.get('key')
        choice = data.get('choice')
        user_question = data.get('question')
        question_key = str(uuid.uuid4())

        if not key or not choice or not user_question:
            print(f"Invalid input data. Key: {key},choice: {choice} user_question: {user_question}")
            logging.error(f"Invalid input data. Key: {key},choice: {choice} user_question: {user_question}")
            return jsonify({"status": 0, "error": "Invalid input data"}), 400

        if len(user_question.split()) <= 2:
            print(f"Question should contain more than 2 words: {user_question}")
            logging.error(f"Question should contain more than 2 words: {user_question}")
            return jsonify({"status": 0, "error": "Question should contain more than 2 words"}), 200

        model_choice = model_mappings.get(choice, 'UNKNOWN_DATABASE')
        d_choice = 'CSG_DATA'

        # Extract relevant data from cached_data
        choice_data = cached_data.get(d_choice, {})
        print('choice data :: ',choice_data)
        all_table_schema_list = choice_data.get('schema_list', [])
        all_table_schema_string = choice_data.get('schema_string', '')
        df = choice_data.get('df', None)
        all_ = choice_data.get('all', None)

        #vectorestore similarity search to get query
        print("pinecone similarity search to get the query if present for asked question")
        print("continuing with open_ai_call flow")
        print(choice)

        start_time_full = time()
        response_1, response_2, final_sample_data, final_ddl_data, roi_tables = open_ai_call(
        question=user_question, model_choice=choice, all_table_schema_string=all_table_schema_string, df=df, all_=all_, all_table_schema_list=all_table_schema_list
        )

        print("finding sql queries....")
        logging.info("finding sql queries....")
        print("printing response 2 again :: \n",response_2)
        queries = find_sql_queries(response_2)
        print("getting running queries....",queries)
        logging.info("getting running queries....")
        compare_queries, df_list = get_running_queries(database_choice, queries)

        #testing
        # compare_queries = []
 
        attempts = 0
 
        while attempts < 3:  # Allow up to 3 attempts
            if len(compare_queries) != 0:
                print("The list is not empty!")
                break
            else:
                print(f"Attempt {attempts + 1}: There are no executable queries.")
                # add the funciton to retry here
                response_1, response_2, final_sample_data, final_ddl_data, roi_tables = open_ai_call(
                question=user_question, model_choice=choice, all_table_schema_string=all_table_schema_string, df=df, all_=all_, all_table_schema_list=all_table_schema_list
                )
 
                print("finding sql queries....")
                logging.info("finding sql queries....")
                print("printing response 2 again :: \n",response_2)
                queries = find_sql_queries(response_2)
                print("getting running queries....",queries)
                logging.info("getting running queries....")
                compare_queries, df_list = get_running_queries(database_choice, queries)
 
                attempts += 1


        print("before find_duplicates_and_unique.. ")
        print("compare_queries : ", compare_queries)
        # print("df_list         : ", df_list)
        print("len of df_list : ", len(df_list))
        logging.info("before find_duplicates_and_unique.. ")
        logging.info("compare_queries : %s", compare_queries)
        # logging.info("df_list         : ", df_list)
        logging.info("len of df_list  : %d ", len(df_list))

        # if df_list:
        if len(df_list)>0:
            duplicates, dfs = find_duplicates_and_unique(df_list)

            print("after find_duplicates_and_unique.. ")
            logging.info("after find_duplicates_and_unique.. ")

            a = [dup[0] for dup in duplicates]
            print("index of duplicate queries  : ", a)
            logging.info("index of duplicate queries  : %s", a)

            final_queries = remove_elements_at_indices(compare_queries, a) if a else compare_queries
            all_queries = final_queries

            print("no. of unique queries ", len(all_queries))
            logging.info("no. of unique queries : %d ", len(all_queries))

            if len(all_queries) > 1:
                print("Ranking the generated queries in correctness order...")
                logging.info("Ranking the generated queries in correctness order...")
                ques, returned_queries = rank_sql_queries(user_question, all_queries, all_table_schema_list, all_,gpt_35_model)
            else:
                print("no ranking needed as we have only one unique query")
                logging.info("no ranking needed as we have only one unique query")
                returned_queries = all_queries[0]

            # returned_queries = find_sql_queries(returned_queries)

            execution_status, df_result = run_query(database_choice, returned_queries)

        # if execution_status == "successfully executed":
        #     if isinstance(df_result, pd.DataFrame):
        #         df_result = df_result.replace({np.nan: None})

        #         # # Add an 'Index' column starting from 1
        #         # df_result = df_result.reset_index(drop=True)  # Reset index
        #         # df_result.index += 1  # Start index from 1
        #         # df_result.insert(0, 'Index', df_result.index)  # Add 'Index' as the first column

        #         df_result = df_result.replace({np.nan: None})

        #         # Properly reset index and drop the old index
        #         df_result = df_result.reset_index(drop=True)
                
        #         # Add an 'Index' column starting from 1
        #         df_result.insert(0, 'Index', range(1, len(df_result) + 1))

        #         print("final query             : ", returned_queries)
        #         print("final query's dataframe : ", df_result)
        #         logging.info("final query             : %s ", returned_queries)

        #         # Prepare the SQL table result
        #         df_array = [df_result.columns.tolist()] + df_result.values.tolist()

        #         # Generate the natural language response
        #         natural_language_response = generate_natural_language_response(user_question, df_result)

        #         # Include both in the JSON response
        #         json_obj = {
        #             "database": database_choice,
        #             "question_key": question_key,
        #             "question": user_question,
        #             "query": returned_queries,
        #             "query_result": df_array,  # SQL table result
        #             "natural_language_response": natural_language_response  # Text response
        #         }
        #         json_response = json.dumps(json_obj, default=default_encoder)
        #         print("json_obj : ", json_obj)
        #         save_chat_to_database(key, json_response)
        #         end_time_full = time()
        #         time_taken_full = round((end_time_full - start_time_full), ndigits=4)
        #         print("Time taken to complete the question :: ", time_taken_full)
        #         return jsonify({"status": 1, "data": json_response})

        if execution_status == "successfully executed":
            if isinstance(df_result, pd.DataFrame):
                # Replace NaN values with None
                df_result = df_result.replace({np.nan: None})

                df = df_result.copy()

                # Step 1: Add the 'Index' column starting from 1
                df.insert(0, 'Index', range(1, len(df) + 1))

                df1 = df.copy()
                

                print("final query             : ", returned_queries)
                # print("final query's dataframe : \n", df_result)
                print("final query's dataframe : \n", df1.to_string(index=False))
                logging.info("final query             : %s ", returned_queries)

                # Prepare the SQL table result
                df_array = [df.columns.tolist()] + df.values.tolist()
                
                print("columns are: ", df.columns.tolist())

                # Generate the natural language response
                natural_language_response = generate_natural_language_response(user_question, df_result)

                # Include both in the JSON response
                json_obj = {
                    "database": database_choice,
                    "question_key": question_key,
                    "question": user_question,
                    "query": returned_queries,
                    "query_result": df_array,  # SQL table result
                    "natural_language_response": natural_language_response  # Text response
                }

                json_response = json.dumps(json_obj, default=default_encoder)
                print("json_obj : ", json_obj)
                print(' ')
                print("json Response: ", json_response)
                save_chat_to_database(key, json_response)
                end_time_full = time()
                time_taken_full = round((end_time_full - start_time_full), ndigits=4)
                print("Time taken to complete the question :: ", time_taken_full)
                return jsonify({"status": 1, "data": json_response})


        else:
            print("did not get any runnable query")
            logging.info("did not get any runnable query")
            json_obj = {
                "database": database_choice,
                "question_key":question_key,
                "question": user_question,
                "query": None,
                "query_result": None,
                # "figure_data": None
            }

            json_response = json.dumps(json_obj, default=default_encoder)
            save_chat_to_database(key, json_response)
            end_time_full = time()
            time_taken_full = round((end_time_full-start_time_full), ndigits=4)
            print("Time taken to complete the question :: ",time_taken_full)
            return jsonify({"status": 1, "data": json_response})

    except Exception as e:
        print(f"An error occurred while processing a POST request at the /chat endpoint  : {e}")
        logging.error(f"An error occurred while processing a POST request at the /chat endpoint  : {e}", exc_info=True)
        end_time_full = time()
        time_taken_full = round((end_time_full-start_time_full), ndigits=4)
        print("Time taken to complete the question :: ",time_taken_full)
        return jsonify({"status": 0, "error": str(e)}), 400

@app.route('/chat-info')
@cross_origin(supports_credentials=True)
def give_chat_details():
    try:
        print("A GET request was received at the /chat-info endpoint.")
        logging.info("A GET request was received at the /chat-info endpoint.")
        output = {'status': 1, 'data': []}
        # Retrieve all documents from the collection
        logging.info("Retrieving chat-history from collection")
        chat_documents = collection.find()

        for chat_document in chat_documents:
            chat_id = str(chat_document['key'])
            question = chat_document['question']
            logging.info(f"Processing chat with ID: {chat_id}, question: {question}")
            output['data'].append({'chatId': chat_id, 'name': question})

        # Logging success
        logging.info("Chat details successfully retrieved")
        return jsonify(output)

    except Exception as e:
        print(f"An error occurred while processing a GET request at the /chat-info endpoint  : {e}")
        logging.error(f"An error occurred while processing a GET request at the /chat-info endpoint  : {e}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}), 400

@app.route('/chat', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_data():
    try:
        key = request.args.get('id')
        print("A GET request was received at the /chat endpoint.")
        logging.info(f"A GET request was received at the /chat endpoint with key: {key}")

        # Find the document with the matching key
        chat_document = collection.find_one({'key': key})

        if chat_document:
            # Convert ObjectId to a string
            chat_document['_id'] = str(chat_document['_id'])
            logging.info("Retrieved document for the provided key")
            return jsonify({"status": 1, "data": chat_document})
        else:
            logging.info("No document found with the provided key")
            return jsonify({"status": 1, "data": {}})

    except Exception as e:
        print(f"An error occurred while processing a GET request at the /chat endpoint  : {e}")
        logging.error(f"An error occurred while processing a GET request at the /chat endpoint  : {e}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}), 400

@app.route('/chat/delete', methods=['GET'])
@cross_origin(supports_credentials=True)
def delete_chat():
    try:
        print("A GET request was received at the /chat/delete endpoint.")
        key = request.args.get('id')
        logging.info(f"A GET request was received at the /chat/delete endpoint to delete chat history for key {key}")

        deleted_document = collection.find_one_and_delete({'key': key})
        if deleted_document:
            logging.info(f"Chat history for key {key} has been deleted!")
            return jsonify({"status": 1, "message": f"Chat history for key {key} has been deleted!"})
        else:
            logging.warning(f"No chat history found for key {key}.")
            return jsonify({"status": 0, "message": f"No chat history found for key {key}."})
    except Exception as e:
        print(f"An error occurred while processing a GET request at the /chat/delete endpoint  : {e}")
        logging.error(f"An error occurred while processing a GET request at the /chat/delete endpoint for key {key}: {str(e)}", exc_info=True)
        return jsonify({"status": 0, "error": str(e)}),400


if __name__ == "__main__":
    app.run(port=5001,host="0.0.0.0")