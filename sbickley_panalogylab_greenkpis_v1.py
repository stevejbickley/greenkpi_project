##### ------ IMPORT FUNCTIONS + SETUP CODE - START ------- ####

from pydantic import BaseModel
from openai import OpenAI
import fitz  # PyMuPDF for PDF handling
import io
import os
from PIL import Image
import base64
import json
import psycopg2
import pandas as pd


##### ------ DEFINE FUNCTIONS - START ------- ####


client = OpenAI()


@staticmethod
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pdf_to_base64_images(pdf_path):
    #Handles PDFs with multiple pages
    pdf_document = fitz.open(pdf_path)
    base64_images = []
    temp_image_paths = []
    total_pages = len(pdf_document)
    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        temp_image_path = f"temp_page_{page_num}.png"
        img.save(temp_image_path, format="PNG")
        temp_image_paths.append(temp_image_path)
        base64_image = encode_image(temp_image_path)
        base64_images.append(base64_image)
    for temp_image_path in temp_image_paths:
        os.remove(temp_image_path)
    return base64_images


# In the extract_invoice_data function, modify the system prompt to extract relevant CV data such as name, contact information, work experience, education, skills, etc.
def extract_invoice_data(base64_image, temperature_setting=0.0, model="gpt-4o-2024-08-06"):
    system_prompt = f"""
    You are an OCR-like data extraction tool that extracts data from documents like bills, invoices, receipts, etc.
    1. Please extract all available information about the document, grouping data into key categories: Buyer Names, Supplier Names, Invoice Numbers, Dates, Location Names, Location Details, Line Usages, Line Cost Per Units, Line Totals, Invoice Totals, Emissions Amounts, Emissions Units.
    2. Please output the data in JSON format with meaningful and consistent keys for each section.
    3. If some sections (e.g., Emissions Amount or Location Details) are missing, include them as "null" values.
    4. Maintain the structure of the resume while grouping similar information together.
    """
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "extract the data from this document and output it into JSON"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        temperature=temperature_setting,
    )
    return response.choices[0].message.content


def extract_from_multiple_pages(base64_images, original_filename, output_directory):
    entire_invoice = []
    for base64_image in base64_images:
        invoice_json = extract_invoice_data(base64_image)
        if not invoice_json:
            continue
        invoice_data = json.loads(invoice_json)
        entire_invoice.append(invoice_data)
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    # Construct the output file path
    output_filename = os.path.join(output_directory, original_filename.replace('.pdf', '_extracted.json'))
    # Save the entire_invoice list as a JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(entire_invoice, f, ensure_ascii=False, indent=4)
    return output_filename


def main_extract(read_path, write_path):
    for filename in os.listdir(read_path):
        if filename[-4:] == '.pdf':
            file_path = os.path.join(read_path, filename)
            if filename.endswith('.pdf'):
                base64_images = pdf_to_base64_images(file_path)
                extract_from_multiple_pages(base64_images, filename, write_path)
            elif filename.endswith(('.jpg', '.png')):
                base64_image = encode_image(file_path)
                extract_invoice_data(base64_image)


# SDK for structured response extraction
class InvoiceAnalyzer(BaseModel):
    Supplier_Name: list[str]
    Invoice_Number: list[str]
    Date: list[str]
    Location_Name: list[str]
    Location_Details: list[str]
    Line_Usage: list[str]
    Line_Cost_Per_Unit: list[str]
    Line_Total: list[str]
    Invoice_Total: list[str]
    Emissions_Amount: list[str]
    Emissions_Unit: list[str]


def transform_data_into_schema(json_raw, json_schema, temperature_setting=0.2, model="gpt-4o-2024-08-06"):
    system_prompt = f"""
    You are a data transformation tool that takes in JSON data and a reference JSON schema, and outputs JSON data according to the schema.
    Not all of the data in the input JSON will fit the schema, so you may need to omit some data or add null values to the output JSON.
    Translate all data into English if not already in English.
    Ensure values are formatted as specified in the schema (e.g. dates as YYYY-MM-DD).
    If a section from the raw JSON closely matches a section in the schema, but doesn't fit exactly, use your judgment to map the data to the closest or most relevant section of the schema.
    For example, if there are sections such as "Total Usage (kWh)" and/or "KWH's" in the raw JSON, map one or both to the "Emissions_Amount" and "Emissions_Unit" fields in the schema.
    Here is the schema:
    {json_schema}
    """
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"Transform the following raw JSON data according to the provided schema. Ensure all data is in English and formatted as specified by values in the schema. Here is the raw JSON: {json_raw}"}
                ]
            }
        ],
        response_format = InvoiceAnalyzer,
        temperature=temperature_setting,
    )
    return json.loads(response.choices[0].message.content)


def main_transform(extracted_invoice_json_path, json_schema, save_path):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    # Process each JSON file in the extracted invoices directory
    for filename in os.listdir(extracted_invoice_json_path):
        if filename.endswith(".json"):
            file_path = os.path.join(extracted_invoice_json_path, filename)
            # Load the extracted JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_raw = json.load(f)
            # Transform the JSON data
            transformed_json = transform_data_into_schema(json_raw, json_schema)
            # Save the transformed JSON to the save directory
            transformed_filename = f"transformed_{filename}"
            transformed_file_path = os.path.join(save_path, transformed_filename)
            with open(transformed_file_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_json, f, ensure_ascii=False, indent=2)


def ingest_transformed_jsons_to_csv_files(json_folder_path, save_folder_path):
    invoice_data = []
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Collect data according to the invoice schema
            invoice_data.append((
                data.get("Supplier_Name", "null"),
                data.get("Invoice_Number", "null"),
                data.get("Date", "null"),
                data.get("Location_Name", "null"),
                data.get("Location_Details", "null"),
                data.get("Line_Usage", "null"),
                data.get("Line_Cost_Per_Unit", "null"),
                data.get("Line_Total", "null"),
                data.get("Invoice_Total", "null"),
                data.get("Emissions_Amount", "null"),
                data.get("Emissions_Unit", "null")
            ))
    # Ensure the save directory exists
    os.makedirs(save_folder_path, exist_ok=True)
    # Save dataset to CSV
    columns = ["Supplier_Name", "Invoice_Number", "Date", "Location_Name",
               "Location_Details", "Line_Usage", "Line_Cost_Per_Unit",
               "Line_Total", "Invoice_Total", "Emissions_Amount", "Emissions_Unit"]
    save_to_csv(invoice_data, columns, os.path.join(save_folder_path, "invoices.csv"))


def ingest_transformed_jsons_to_wide_csv(json_folder_path, save_file_path):
    all_data = []
    max_line_usage = 0  # To track the maximum number of line usages for dynamic columns
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Collect data according to the invoice schema
            invoice_data = {
                "Supplier_Name": data.get("Supplier_Name", "null"),
                "Invoice_Number": data.get("Invoice_Number", "null"),
                "Date": data.get("Date", "null"),
                "Location_Name": data.get("Location_Name", "null"),
                "Location_Details": data.get("Location_Details", "null"),
                "Invoice_Total": data.get("Invoice_Total", "null"),
                "Emissions_Amount": data.get("Emissions_Amount", "null"),
                "Emissions_Unit": data.get("Emissions_Unit", "null")
            }
            # Handling multiple lines for Line_Usage and associated details
            for i, usage in enumerate(data.get("Line_Usage", []), start=1):
                invoice_data[f"Line_Usage_{i}"] = usage
                invoice_data[f"Line_Cost_Per_Unit_{i}"] = data.get("Line_Cost_Per_Unit", [])[i - 1] if len(
                    data.get("Line_Cost_Per_Unit", [])) > i - 1 else "null"
                invoice_data[f"Line_Total_{i}"] = data.get("Line_Total", [])[i - 1] if len(
                    data.get("Line_Total", [])) > i - 1 else "null"

            max_line_usage = max(max_line_usage, len(data.get("Line_Usage", [])))
            all_data.append(invoice_data)

    # Dynamically create column headers based on max number of line items
    columns = ["Supplier_Name", "Invoice_Number", "Date", "Location_Name", "Location_Details", "Invoice_Total",
               "Emissions_Amount", "Emissions_Unit"]
    for i in range(1, max_line_usage + 1):
        columns += [f"Line_Usage_{i}", f"Line_Cost_Per_Unit_{i}", f"Line_Total_{i}"]
    # Save the data to a wide CSV
    save_to_csv(all_data, columns, save_file_path)


def ingest_transformed_jsons_postgres(json_folder_path, db_config):
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    # Create necessary tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Invoices (
        invoice_id SERIAL PRIMARY KEY,
        supplier_name VARCHAR(255),
        invoice_number VARCHAR(255),
        date DATE,
        location_name VARCHAR(255),
        location_details TEXT,
        invoice_total FLOAT,
        emissions_amount FLOAT,
        emissions_unit VARCHAR(50)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS LineItems (
        line_id SERIAL PRIMARY KEY,
        invoice_id INTEGER REFERENCES Invoices(invoice_id),
        line_usage TEXT,
        line_cost_per_unit FLOAT,
        line_total FLOAT
    )
    ''')
    # Loop over all JSON files and insert data
    for filename in os.listdir(json_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Insert Invoice Data
            cursor.execute('''
            INSERT INTO Invoices (supplier_name, invoice_number, date, location_name, location_details, invoice_total, emissions_amount, emissions_unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING invoice_id
            ''', (
                data.get("Supplier_Name", None),
                data.get("Invoice_Number", None),
                data.get("Date", None),
                data.get("Location_Name", None),
                data.get("Location_Details", None),
                data.get("Invoice_Total", None),
                data.get("Emissions_Amount", None),
                data.get("Emissions_Unit", None)
            ))
            invoice_id = cursor.fetchone()[0]
            # Insert Line Items
            for i, line_usage in enumerate(data.get("Line_Usage", [])):
                line_cost_per_unit = data.get("Line_Cost_Per_Unit", [])[i] if i < len(
                    data.get("Line_Cost_Per_Unit", [])) else None
                line_total = data.get("Line_Total", [])[i] if i < len(data.get("Line_Total", [])) else None
                cursor.execute('''
                INSERT INTO LineItems (invoice_id, line_usage, line_cost_per_unit, line_total)
                VALUES (%s, %s, %s, %s)
                ''', (invoice_id, line_usage, line_cost_per_unit, line_total))
    # Commit changes and close connection
    conn.commit()
    cursor.close()
    conn.close()


def main_pipeline(input_folders, json_schema, output_folder):
    """Main pipeline to extract and save CSV data."""
    for folder in input_folders:
        read_path = os.path.join('./data/input', folder)
        # Step 1: Extract data
        main_extract(read_path, output_folder)
        # Step 2: Transform data into CSV
        #main_transform(intermediate_path, json_schema, output_folder)


##### ------ MAIN CODE - START ------- ####

# -- Step 1)

# Define the schema inline for structured response extraction
invoice_schema = {
    "Supplier_Name": "string",
    "Invoice_Number": "string",
    "Date": "YYYY-MM-DD",
    "Location_Name": "string",
    "Location_Details": "string",
    "Line_Usage": "string",
    "Line_Cost_Per_Unit": "float",
    "Line_Total": "float",
    "Invoice_Total": "float",
    "Emissions_Amount": "float",
    "Emissions_Unit": "string"
}

# List of input folders
input_folders = ['Annotated_bills', 'electricity_invoice', 'fuel_receipts']
output_folder = './data/output/intermediate'
main_pipeline(input_folders, InvoiceAnalyzer, output_folder)



# -- Step 2)

intermediate_path = output_folder
output_folder = './data/output/intermediate/transformed/'
main_transform(intermediate_path, InvoiceAnalyzer, output_folder)

# -- Step 2)

# Get database connection details from environment variables
db_config = {
    "dbname": "greenkpi",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Read in the jsons and ingest/push to postgres
ingest_transformed_jsons_postgres(output_folder, db_config)

# Example usage with save to csv files instead
#save_folder_path = "./data/outputs/"
#ingest_transformed_jsons_to_csv_files(json_folder_path, save_folder_path)

# Example usage
save_file_path = "./data/outputs/wide_resumes_output.csv"
ingest_transformed_jsons_to_wide_csv(output_folder, save_file_path)


##### ------ MAIN CODE - END ------- ####

