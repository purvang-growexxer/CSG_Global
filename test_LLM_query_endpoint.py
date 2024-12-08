from flask import Flask, request, jsonify
import pandas as pd
import snowflake.connector
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
from pymongo import MongoClient
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


# Initialize Flask app
app = Flask(__name__)

# Initialize the Groq client with the API key
g_llm = Groq(
    api_key=groq_ai_key,  # Pass the API key explicitly
)

def query_to_display_all_columns(input_query):

    try:

        # Define the prompt with the given input_query
        prompt = f"""
        You are an expert in SQL. Below is a query that returns filtered rows and selected columns. 
        Your task is to modify the query so that only the filtered rows are returned, 
        but with all columns included in the output. Ensure that the updated query is syntactically correct, 
        executes without errors, and provides the desired results.

        These are the column details for your reference:
        MATCHED_PO_AMOUNT: The total value of the purchase order matched with invoices or goods/services received.
        PO_CURRENCY: The currency used in the purchase order transaction.
        PO_END_DATE: The final date for fulfilling the purchase order.
        INVOICE_APPROVAL_STATUS: The current status of the invoice approval process.
        PO_REQUESTER_SUPERVISOR: The supervisor of the person who requested the purchase order.
        CURRENT_APPROVER: The individual currently responsible for approving the purchase order or invoice.
        PO_REQUESTER: The person who requested the purchase order.
        PO_NUMBER: The unique identifier assigned to the purchase order.
        INVOICE_AMOUNT: The total amount stated on the invoice for goods or services provided.
        OVERALL_PO_CONSUMED_AMOUNT: The amount of the purchase order that has been used or consumed.
        INVOICE_VALIDATION_STATUS: The status of the invoice's validation process.
        CURRENT_APPROVER_END_DATE: The deadline by which the current approver should make a decision.
        INVOICE_NUMBER: The unique identifier assigned to the invoice.
        PAYMENT_STATUS: The status of the payment for the invoice (e.g., paid, pending).
        SUPPLIER_NUMBER: The unique identifier for the supplier.
        PO_UNITPRICE: The price per unit of the items or services in the purchase order.
        PO_REQUESTER_EMAIL: The email address of the person who requested the purchase order.
        PO_REQ_END_DATE: The expected end date for the requested purchase order items.
        BUSINESS_UNIT_NAME: The name of the business unit that requested the purchase order.
        PO_ORIGINAL_AMOUNT: The original approved amount of the purchase order.
        OVERALL_PO_AMOUNT_REMAINING: The remaining balance of the purchase order after consumption.
        HOLD_STATUS: Indicates if the purchase order or invoice is on hold.
        MATCHED_PO_QUANTITY: The quantity of items from the purchase order matched with invoices or deliveries.
        SUPPLIER_NAME: The name of the supplier or vendor providing the goods or services.

        {input_query} 

        INSTRUCTIONS:
        Modify the query provided above such that:
        1. The output includes all columns.
        2. Only filtered rows are retained.
        3. Ensure the syntax is correct, and no additional text, explanations, or symbols are included in the response.
        Only return the modified query without any other information.

        """

        # Debugging: Uncomment to print the prompt and see the generated text
        # print(prompt)
 
        # Sending the prompt to the LLM for processing
        response = g_llm.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": "You are an expert in SQL"},
                {"role": "user", "content": prompt}
            ]
        )

        # Return the content of the response
        return response.choices[0].message.content
 
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {e}")
        return ["Could not generate follow-up questions due to an error."]

def default_encoder(obj):
    if isinstance(obj, (datetime, date, ObjectId)):
        return str(obj)
    raise TypeError("Object not serializable")


# Function to execute query in Snowflake
def run_query(database, query):
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
        return "successfully executed", df
    except snowflake.connector.Error as e:
        print(f"Snowflake Error: {e}")
        return "error while executing", pd.DataFrame()
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return "error while executing", pd.DataFrame()

def update_data():
    data = request.get_json()
    key = data.get("key")
    question_key = data.get("question_key")
    modified_query = data.get("modified_all_column_query")
    df_array = data.get("all_columns_result")

    if not (key and question_key and modified_query and df_array):
        return jsonify({"error": "Invalid input data"}), 400

    # Update MongoDB document
    result = collection.update_one(
        {"key": key, "question_key": question_key},
        {
            "$set": {
                "modified_all_column_query": modified_query,
                "all_columns_result": df_array,
            }
        }
    )

    if result.matched_count > 0:
        return jsonify({"message": "Document updated successfully"}), 200
    else:
        return jsonify({"error": "No document found with the given keys"}), 404

# Flask endpoint
@app.route('/hyperlink', methods=['POST'])
def execute_query():
    try:
        # Parse input data
        data = request.json
        print(data)
        question_key = data.get('question_key')
        key = data.get('chat_key')
        # input_query = data.get('input_query')
        database = mongo_database
        
        # Connect to the MongoDB server
        client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB URI

        # Access the database and collection
        client = MongoClient(mongo_conn_string) # original app.py mai hai. To iss line ki jarurat nai hai. 
        db = client[mongo_database]
        collection = db[mongo_collection]

        # Query to retrieve the document
        document = collection.find_one({"key": key, "question_key": question_key}, {"_id": 0, "query": 1})

        if document:
            query_value = document.get("query", None)
            print("Query retrieved:", query_value)
        else:
            print("No matching document found.")


        if not query_value or not database:
            return jsonify({"error": "Both 'query' and 'database' are required"}), 400

        # Generate follow-up query
        modified_query = query_to_display_all_columns(query_value)
        if not modified_query:
            return jsonify({"error": "Failed to generate modified query"}), 500

        # Execute the modified query
        # status, result_df = run_query(database, modified_query)
        execution_status, df_result = run_query(database, modified_query)
        if execution_status == "successfully executed":
            # Step 1: Add the 'Index' column starting from 1
            df_result = df_result.replace({np.nan: None})

            df = df_result.copy()
            df.insert(0, 'Index', range(1, len(df) + 1))

            df1 = df.copy()

            # print("updated query             : ", modified_query)
            # print("Updated query's dataframe : ", df1.to_string(index=False))
            # logging.info("final query             : %s ", modified_query)
            df_array = [df.columns.tolist()] + df.values.tolist()
            # print("columns are: ", df.columns.tolist())

            json_obj = {
                "key": key,
                "question_key": question_key,
                "modified_query": modified_query,
                "all_columns_result": df_array,
            }

            # Update MongoDB document
            update_result = collection.update_one(
                {"key": key, "question_key": question_key},
                {
                    "$set": {
                        "modified_all_column_query": modified_query,
                        "all_columns_result": df_array,
                    }
                }
            )

            if update_result.matched_count > 0:
                print("MongoDB document updated successfully.")
            else:
                print("No document found to update.")

            json_response = json.dumps(json_obj, default=default_encoder)
            print("json_obj : ", json_obj)
            return jsonify({"status": 1, "data": json_response})


    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

