import openai
import requests
from openai import OpenAI
import csv
import PyPDF2
import re
import os
import requests
import pandas as pd
import tiktoken
import time
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
import ast

def extract_highlight(pdf_file):
    # 打开PDF文件
    pdf_file = open(pdf_file, 'rb')

    # 读取PDF文件对象
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # 获取PDF文件的第一页
    
    # 获取页面中的所有高亮文本
    for page in pdf_reader.pages:
        try:
            highlights = [annot for annot in page['/Annots'] if annot.get_object()['/Subtype'] == '/Highlight']

            # 遍历所有高亮文本并输出其内容
            for highlight in highlights:
                print(highlight.get_object().keys())#['/Contents'])
        except:
            pass
    pdf_file.close()
    return highlights


def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def get_txt_from_pdf(pdf_files,filter_ref = False, combine=False):
    """Convert pdf files to dataframe"""
    # Create an empty list to store the data
    data = []
    # Iterate over the PDF
    #print(pdf_files)
    for pdf in pdf_files:
        # Fetch the PDF content from the pdf
        with open(pdf, 'rb') as pdf_content:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            # Iterate over all the pages in the PDF
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num] # Extract the text from the current page
                page_text = page.extract_text()
                words = page_text.split() # Split the page text into individual words
                page_text_join = ' '.join(words) # Join the words back together with a single space between each word

                if filter_ref: #filter the reference at the end
                    page_text_join = remove_ref(page_text_join)

                page_len = len(page_text_join)
                div_len = page_len // 4 # Divide the page into 4 parts
                page_parts = [page_text_join[i*div_len:(i+1)*div_len] for i in range(4)]
            
                min_tokens = 40
                for i, page_part in enumerate(page_parts):
                    if count_tokens(page_part) > min_tokens:
                        # Append the data to the list
                        data.append({
                            'file name': pdf,
                            'page number': page_num + 1,
                            'page section': i+1,
                            'content': page_part,
                            'tokens': count_tokens(page_part)
                        })
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    if combine:
        df = combine_section(df)
    return df


def remove_ref(pdf_text):
    """This function removes reference section from a given PDF text. It uses regular expressions to find the index of the words to be filtered out."""
    # Regular expression pattern for the words to be filtered out
    pattern = r'(REFERENCES|Acknowledgment|ACKNOWLEDGMENT)'
    match = re.search(pattern, pdf_text)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        clean_text = pdf_text[:start_index].strip()
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            '\[[\d\w]{1,3}\].+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5};','\([\d\w]{1,3}\).+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5},',
            '\([\d\w]{1,3}\).+?[\d]{3,5},','\[[\d\w]{1,3}\].+?[\d]{3,5}','[\d\w]{1,3}\).+?[\d]{3,5}\.','[\d\w]{1,3}\).+?[\d]{3,5}',
            '\([\d\w]{1,3}\).+?[\d]{3,5}','^[\w\d,\.– ;)-]+$',
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = re.sub(pattern, '', pdf_text) if len(matches) > 500 and matches.count('.') < 2 and matches.count(',') < 2 and not matches[-1].isdigit() else pdf_text

        # Split the text into lines
        lines = pdf_text.split('\n')

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = re.sub(pattern, '', lines[i]) if len(matches) > 500 and len(re.findall('\d', matches)) < 8 and len(set(matches)) > 10 and matches.count(',') < 2 and len(matches) > 20 else lines[i]

        # Join the lines back together, excluding any empty lines
        clean_text = '\n'.join([line for line in lines if line])

    return clean_text

      
def combine_section(df):
    """Merge sections, page numbers, add up content, and tokens based on the pdf name."""
    aggregated_df = df.groupby('file name').agg({
        'content': aggregate_content,
        'tokens': aggregate_tokens
    }).reset_index()

    return aggregated_df


def aggregate_content(series):
    """Join all elements in the series with a space separator. """
    return ' '.join(series)


def aggregate_tokens(series):
    """Sum all elements in the series."""
    return series.sum()


def extract_title(file_name):
    """Extract the main part of the file name. """
    title = file_name.split('_')[0]
    return title.rstrip('.pdf')


def combine_main_SI(df):
    """Create a new column with the main part of the file name, group the DataFrame by the new column, 
    and aggregate the content and tokens."""
    df['main_part'] = df['file name'].apply(extract_title)
    merged_df = df.groupby('main_part').agg({
        'content': ''.join,
        'tokens': sum
    }).reset_index()

    return merged_df.rename(columns={'main_part': 'file name'})


def df_to_csv(df, file_name):
    """Write a DataFrame to a CSV file."""
    df.to_csv(file_name, index=False, escapechar='\\')


def csv_to_df(file_name):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_name)



def tabulate_condition(df,column_name):
    """This function converts the text from a ChatGPT conversation into a DataFrame.
    It also cleans the DataFrame by dropping additional headers and empty lines.    """
    
    table_text = df[column_name].str.cat(sep='\n')

    # Remove leading and trailing whitespace
    table_text = table_text.strip()
    
    # Split the table into rows
    rows = table_text.split('\n')

    # Extract the header row and the divider row
    header_row, divider_row, *data_rows = rows

    # Extract column names from the header row
    #| Polymer monomers | initiators | temperature | time | Free radical polymerization mechanism, cationic and anionic ring-opening polymerization | Conductivity | ionic transference number | electrochemical window | critical current density (CCD) |  tensile strength of the polymer
    column_names = ['polymer monomers', 'initiators', 'temperature', 'time', 'Free radical polymerization mechanism, cationic and anionic ring-opening polymerization', 'conductivity', 'ionic transference number', 'electrochemical window', 'critical current density (CCD)', 'tensile strength of the polymer']
    '''column_names = ['compound name', 'metal source', 'metal amount', 'linker', 'linker amount',
                   'modulator', 'modulator amount or volume', 'solvent', 'solvent volume', 'reaction temperature',
                   'reaction time']'''

    # Create a list of dictionaries to store the table data
    data = []

    # Process each data row
    for row in data_rows:

        # Split the row into columns
        columns = [col.strip() for col in row.split('|') if col.strip()]
    
        # Create a dictionary to store the row data
        row_data = {col_name: col_value for col_name, col_value in zip(column_names, columns)}
    
        # Append the dictionary to the data list
        data.append(row_data)
        
    df = pd.DataFrame(data)
    # 2024.4.8 test
    return df
    
    """Make df clean by drop additional header and empty lines """
    def contains_pattern(s, patterns):
        return any(re.search(p, s) for p in patterns)

    def drop_rows_with_patterns(df, column_name):
        #empty cells, N/A cells and header cells
        patterns = [r'^\s*$', r'--',r'-\s-', r'compound', r'Compound',r'Compound name', r'Compound Name',
                r'NaN',r'N/A',r'n/a',r'\nN/A', r'note', r'Note']
        
        mask = df[column_name].apply(lambda x: not contains_pattern(str(x), patterns))
        filtered_df = df[mask]
    
        return filtered_df
    
    
    #drop the repeated header
    df = drop_rows_with_patterns(df, 'compound name')
    
    #drop the organic synthesis (where the metal source is N/a)    
    filtered_df = drop_rows_with_patterns(drop_rows_with_patterns(drop_rows_with_patterns(df,'metal source'),'metal amount'),'linker amount') 

    #drop the N/A rows
    filtered_df = filtered_df.dropna(subset=['metal source','metal amount', 'linker amount'])

    return filtered_df



def split_content(input_string, tokens):
    """Splits a string into chunks based on a maximum token count. """

    MAX_TOKENS = tokens
    split_strings = []
    current_string = ""
    tokens_so_far = 0

    for word in input_string.split():
        # Check if adding the next word would exceed the max token limit
        if tokens_so_far + count_tokens(word) > MAX_TOKENS:
            # If we've reached the max tokens, look for the last dot or newline in the current string
            last_dot = current_string.rfind(".")
            last_newline = current_string.rfind("\n")

            # Find the index to cut the current string
            cut_index = max(last_dot, last_newline)

            # If there's no dot or newline, we'll just cut at the max tokens
            if cut_index == -1:
                cut_index = MAX_TOKENS

            # Add the substring to the result list and reset the current string and tokens_so_far
            split_strings.append(current_string[:cut_index + 1].strip())
            current_string = current_string[cut_index + 1:].strip()
            tokens_so_far = count_tokens(current_string)

        # Add the current word to the current string and update the token count
        current_string += " " + word
        tokens_so_far += count_tokens(word)

    # Add the remaining current string to the result list
    split_strings.append(current_string.strip())

    return split_strings


def table_text_clean(text):
    """Cleans the table string and splits it into lines."""

    # Pattern to find table starts
    pattern = r"\|\s*compound\s*.*"

    # Use re.finditer() to find all instances of the pattern in the string and their starting indexes
    matches = [match.start() for match in re.finditer(pattern, text, flags=re.IGNORECASE)]

    # Count the number of matches
    num_matches = len(matches)

    # Base table string
    table_string = """| compound name | metal source | metal amount | linker | linker amount | modulator | modulator amount or volume | solvent | solvent volume | reaction temperature | reaction time |\n|---------------|-------|--------------|--------|---------------|-----------|---------------------------|---------|----------------|---------------------|---------------|\n"""

    if num_matches == 0:  # No table in the answer
        print("No table found in the text: " + text)
        splited_text = ''

    else:  # Split the text based on header
        splited_text = ''
        for i in range(num_matches):
            # Get the relevant table slice
            splited = text[matches[i]:matches[i + 1]] if i != (num_matches - 1) else text[matches[i]:]

            # Remove the text after last '|'
            last_pipe_index = splited.rfind('|')
            splited = splited[:last_pipe_index + 1]

            # Remove the header and \------\
            pattern_dash = r"-(\s*)\|"
            match = max(re.finditer(pattern_dash, splited), default=None, key=lambda x: x.start())

            if not match:
                print("'-|' pattern not found.")
            else:
                first_pipe_index = match.start()
                splited = '\n' + splited[(first_pipe_index + len('-|\n|') - 1):]  # Start from "\"

            splited_text += splited

    table_string = table_string + splited_text
    return table_string

def add_similarity(df, given_embedding):
    """Adds a 'similarity' column to a dataframe based on cosine similarity with a given embedding."""
    def calculate_similarity(embedding):
        # Check if embedding is a string and convert it to a list of floats if necessary
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip('[]').split(',')]
        return cosine_similarity([embedding], [given_embedding])[0][0]

    df['similarity'] = df['embedding'].apply(calculate_similarity)
    return df


def select_top_neighbors(df):
    """Retains top-10 similarity sections and their neighbors in the dataframe and drops the rest."""
    # Sort dataframe by 'file name' and 'similarity' in descending order
    df.sort_values(['file name', 'similarity'], ascending=[True, False], inplace=True)
    
    # Group dataframe by 'file name' and select the top 10 rows based on similarity
    top_10 = df.groupby('file name').head(10)
    
    # Add neighboring rows (one above and one below) to the selection
    neighbors = [i for index in top_10.index for i in (index - 1, index + 1) if 0 <= i < df.shape[0]]

    # Create a new dataframe with only the selected rows
    selected_df = df.loc[top_10.index.union(neighbors)]
    return selected_df


def add_emb(df):
    """Adds an 'embedding' column to a dataframe using OpenAI API."""
    
    if 'embedding' in df.columns:
        print('The dataframe already has embeddings. Please double check.')
        return df

    embed_msgs = []
    for _, row in df.iterrows():
        context = row['content']
        context_emb = client.embeddings.create(model="text-embedding-ada-002", input=context)
        #print(context_emb)
        embed_msgs.append(context_emb.data[0].embedding)

    df = df.copy()
    df.loc[:, 'embedding'] = embed_msgs
    
    return df

   

def model_1(df):
    """Model 1 will turn text in dataframe to a summarized reaction condition table.The dataframe should have a column "file name" and a column "exp content"."""
    response_msgs = []

    for index, row in df.iterrows():
        column1_value = row[df.columns[0]]
        column2_value = row['content']

        max_tokens = 3000
        if count_tokens(column2_value) > max_tokens:
            context_list = split_content(column2_value, max_tokens)
        else:
            context_list = [column2_value]

        answers = ''  # Collect answers from chatGPT
        #polymer electrolyte
        #Experimental Materials: Polymer monomers, initiators, temperature, time
        #Polymerization Mechanism: (Free radical polymerization mechanism, cationic and anionic ring-opening polymerization) 
        #Properties of Interest: Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer.
        for context in context_list:
            print("Start to analyze paper " + str(column1_value) )
            user_heading = f"This is an experimental section on polymer electrolyte from paper {column1_value}\n\nContext:\n{context}"
            user_ending = """Q: Can you summarize the following details in a table: 
            1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            If any information is not provided or you are unsure, use "N/A". 
            Please focus on extracting experimental conditions from only the polymer electrolyte. 
            If multiple conditions are provided for the same compound, use multiple rows to represent them. If multiple units or components are provided for the same factor (e.g.  g and mol for the weight, multiple linker or metals, multiple temperature and reaction time, mixed solvents, etc), include them in the same cell and separate by comma.
            The table should have 11 columns, all in lowercase:
            | Polymer monomers | initiators | temperature | time | Free radical polymerization mechanism, cationic and anionic ring-opening polymerization | Conductivity | ionic transference number | electrochemical window | critical current density (CCD) |  tensile strength of the polymer
            
            A:"""   

            attempts = 3
            while attempts > 0:
                try:
                    response = client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=[{
                            "role": "system",
                            "content": """Answer the question as truthfully as possible using the provided context,
                                        and if the answer is not contained within the text below, say "N/A" """
                        },
                            {"role": "user", "content": user_heading + user_ending}]
                    )
                    answer_str = response.choices[0].message.content
                    if not answer_str.lower().startswith("n/a"):
                        answers += '\n' + answer_str
                    break
                except Exception as e:
                    attempts -= 1
                    if attempts <= 0:
                        print(f"Error: Failed to process paper {column1_value}. Skipping. (model 1)")
                        break
                    print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
                    time.sleep(60)

        response_msgs.append(answers)
    df = df.copy()
    df.loc[:, 'summarized'] = response_msgs
    return df


def model_2(df):
    """Model 2 has two parts. First, it asks ChatGPT to identify the experiment section,
    then it combines the results"""
    
    response_msgs = []
    
    prev_paper_name = None  # Initialize the variable. For message printing purpose
    total_pages = df.groupby(df.columns[0])[df.columns[1]].max() #  For message printing purpose
    
    for _, row in df.iterrows():
        paper_name = row[df.columns[0]]
        page_number = row[df.columns[1]]
        # Only print the message when the paper name changes
        if paper_name != prev_paper_name:
            print(f'Processing paper: {paper_name}. Total pages: {total_pages[paper_name]}')
            prev_paper_name = paper_name

        context = row['content']

        user_msg1 = """
        Context:
        The ionic conductivity of PVCA-SPE is 2.23 × 10−5 S cm−1 at 25 °C and 9.82 × 10−5 S cm−1 at 50 °C, which is higher than that of previous reported PEO based electrolyte and PAN based electrolyte.
        Question: Does the section contain one of these: 
        1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
        2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
        3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: Yes.

        Context:
        To fully utilize the advantages of FP cross-linker, vinylene carbonate (VC) has been selected as the monomer to fabricate FP-GPE, ensuring superior interfacial compatibility with both high-voltage cathodes and LMA. 
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: Yes.

        Context:
        the polymer matrix of FP-GPE exhibits a low HOMO energy level (−7.64 eV) compared to ETPTA cross-linked GPE (C-GPE) (−6.43 eV) and TAEP cross-linked GPE (P-GPE) (−7.39 eV)
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: Yes.

        Context:
        As shown in Figure 4d, the current density began to increase obviously at 4.5 V versus Li/Li+ at 50 °C.
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: Yes.

        Context:
        After heating process at 60 °C for 24 h and 80 °C for 10 h, the pouch type battery was charged to 4.3 V at 50 °C. Then, the battery endured six consecutive nail penetration tests. It is worthwhile to note that the battery kept a good shape without any flame and explosion and displayed a relative high voltage at 4.02 V.
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: Yes.
        
        Context:
        Nowadays it is extremely urgent to seek high performance solid polymer electrolyte that possesses both interfacial stability toward lithium/graphitic anodes and high voltage cathodes for high energy density solid state batteries. 
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: No.

        Context:
        In addition, there was no observable short circuit phenomenon after 600 h polarization at both 0.05 and 0.10 mA cm−2. 
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: No.
        
        Context:
        Thermogravimetric (TG) curves and flame tests (Figure 2g,h) demonstrate that CPCE loses just 5 percent of its pristine weight at 168 °C and cannot be ignited even in contact with the flame for several seconds, showing low vapor pressure and nonflammability properties. 
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: No.
        
        Context:
        Whenthe current density was increased to 2.5 mA cm2 and 3mAcm2 (Fig. 4b and Fig. S16, ESI†), the Li||Li cell using either L-DOX electrolyte or P-DOL PE is not able to cycle normally and the cell fails within a few cycles
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer: No.
                
        Context:
          """
    
        user_msg2 = """
        Question: Does the section contain one of these: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
            
        Answer:
        """

        attempts = 3
        while attempts > 0:
            try:
                
                

                
                response = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": """Determine if the section comes from an section for polymer electrolyte, which contains information on at least one of the following: 1. Experimental Materials, such as Polymer monomers, initiators, temperature, time. 
                            2. Polymerization Mechanism, such as(Free radical polymerization mechanism, cationic and anionic ring-opening polymerization).
                            3. Properties of Interest, such as Conductivity, ionic transference number, electrochemical window, critical current density (CCD), tensile strength of the polymer? 
                            . Answer will be either Yes or No."""},
                        {"role": "user", "content": user_msg1 + context + user_msg2}
                    ]
                )
                answers = response.choices[0].message.content
                break

            except Exception as e:
                attempts -= 1
                if attempts > 0:
                    print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 2)")
                    time.sleep(60)
                else:
                    print(f"Error: Failed to process paper {paper_name}. Skipping. (model 2)")
                    answers = "No"
                    break

        response_msgs.append(answers)
    df = df.copy()
    df.loc[:,'classification'] = response_msgs


    # The following section creates a new dataframe after applying some transformations to the old dataframe
    # Create a boolean mask for rows where 'results' starts with 'No'
    mask_no = df["classification"].str.startswith("No")
    # Create a boolean mask for rows where both the row above and below have 'No' in the 'results' column
    mask_surrounded_by_no = mask_no.shift(1, fill_value=False) & mask_no.shift(-1, fill_value=False)
    # Combine the two masks with an AND operation
    mask_to_remove = mask_no & mask_surrounded_by_no
    # Invert the mask and filter the DataFrame
    filtered_df = df[~mask_to_remove]
    #combined
    combined_df= combine_main_SI(combine_section(filtered_df ))
    #call model 1 to summarized results
    add_table_df = model_1(combined_df)
    return add_table_df 


def model_3(df, prompt_choice="synthesis", classfication = True):
    """Input a dataframe in broken separation, ~300 tokens, separated by pages and sections. This function will filter the unnecessary sections."""

    # Set up your API key
    openai.api_key = api_key

    # Define the prompt
    prompts = {
        "synthesis": "Provide a detailed description of the experimental section or synthesis method used in this research. This section should cover essential information such as the compound name (e.g., MOF-5, ZIF-1, Cu(Bpdc), compound 1, etc.), metal source (e.g., ZrCl4, CuCl2, AlCl3, zinc nitrate, iron acetate, etc.), organic linker (e.g., terephthalate acid, H2BDC, H2PZDC, H4Por, etc.), amount (e.g., 25mg, 1.02g, 100mmol, 0.2mol, etc.), solvent (e.g., N,N Dimethylformamide, DMF, DCM, DEF, NMP, water, EtOH, etc.), solvent volume (e.g., 12mL, 100mL, 1L, 0.1mL, etc.), reaction temperature (e.g., 120°C, 293K, 100C, room temperature, reflux, etc.), and reaction time (e.g., 120h, 1 day, 1d, 1h, 0.5h, 30min, a week, etc.).",
        "TGA": """Identify the section discussing thermogravimetric analysis (TGA) and thermal stability. This section typically includes information about weight-loss steps (e.g., 20%, 30%, 29.5%) and a decomposition temperature range (e.g., 450°C, 515°C) or a plateau.""",
        "sorption": "Identify the section discussing nitrogen (N2) sorption, argon sorption, Brunauer-Emmett-Teller (BET) surface area, Langmuir surface area, and porosity. This section typically reports values such as 1000 m2/g, 100 cm3/g STP, and includes pore diameter or pore size expressed in units of Ångströms (Å)"
    }
        
    #other than "synthesis", "TGA", "sorption"),the prompt choice can be the name of the linker to be searched for.
    # If the choice is not one of the predefined ones ("synthesis", "TGA", "sorption"), it defaults to a generic prompt for the linker.
    prompt = prompts.get(prompt_choice, f"Provide the full name of linker ({prompt_choice}) or denoted as {prompt_choice} in chemicals, abstract, introduction or experimental section.")
    
    # Create an embedding for the chosen prompt using OpenAI's embedding model
    prompt_result = client.embeddings.create(model="text-embedding-ada-002", input=prompt)
    # Extract the embedding data from the result
    prompt_emb = prompt_result['data'][0]['embedding']

    # If the dataframe does not already have an 'embedding' column, add one. This is done by calling the add_emb function on the dataframe
    if 'embedding' not in df.columns:
        df_with_emb = add_emb(df)
    else:
        df_with_emb  = df

    # Add a 'similarity' column to the dataframe by comparing the embeddings.This is done by calling the add_similarity function on the dataframe and the prompt embedding
    df_2 = add_similarity(df_with_emb, prompt_emb)

    # Filter the dataframe to only include rows with top similarity and their neighbors
    df_3 = select_top_neighbors(df_2)

    # If the classification parameter is True, pass the dataframe to model_2 for further processing
    if classfication:
        return model_2(df_3)

    # If the classification parameter is False, return the filtered dataframe as is
    return df_3



def load_paper(filename):
    """Crate a dataframe"""
    if os.path.exists(filename):
        dataframe = pd.read_csv(filename,encoding='unicode_escape')
        return dataframe
    else:
        #load pdf names
        
        with open('pdf_pool.csv', 'r') as file:
            reader = csv.reader(file)
            pdf_pool = [row[0] for row in reader]
        dataframe = get_txt_from_pdf(pdf_pool,combine = False, filter_ref = True)
    
        #store the dataframe
        df_to_csv(dataframe, filename)

        
def load_paper_emb(filename):
    """Crate a dataframe that includes embedding information"""
    if os.path.exists(filename):
        paper_df_emb  = pd.read_csv(filename,encoding='unicode_escape')
        paper_df_emb['embedding'] = paper_df_emb['embedding'].apply(ast.literal_eval)
        
    else: #load paper and create embedding
        paper_df_emb = add_emb(load_paper(filename))
    #store embedding to csv
        df_to_csv(paper_df_emb, filename)
    
    return paper_df_emb


def check_system(syn_df, paper_df, paper_df_emb):
    """Check if the data is correctly loaded"""
    # check if openai.api_key is not placeholder
    if openai.api_key  == "Add Your OpenAI API KEY Here.":
        print("Error: Please replace openai.api_key with your actual key.")
        return False

    # check if 'content' column exists in syn_df
    if 'content' not in syn_df.columns:
        print("Error: 'content' column is missing in syn_df.")
        return False

    # check if 'paper_df' has at least four columns
    expected_columns = ['file name', 'page number', 'page section', 'content']
    if not all(col in paper_df.columns for col in expected_columns):
        print("Error: 'paper_df' should have these columns: 'file name', 'page number', 'page section', 'content'.")
        return False

    # check if 'embedding' column exists in paper_df_emb
    if 'embedding' not in paper_df_emb.columns:
        print("Error: 'embedding' column is missing in paper_df_emb.")
        return False

    print("All checks passed.")
    return True



#Load all dataframes
if __name__=='__main__': #e.g. openai.api_key = "abcdefg123abc" 
    


    gpt4='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'
    claude='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'

    client = OpenAI(
    api_key=gpt4,  # this is also the default, it can be omitted
    )
    
    '''
    syn_df = pd.read_csv("228paper_info.csv")
    paper_df=load_paper("228paper_parsed.csv")
    #paper_df_emb = load_paper_emb("228paper_emb.csv")
    #check_system(syn_df, paper_df, paper_df_emb)

    #Run for Model 1
    model_1_table = tabulate_condition(model_1(syn_df),"summarized")
    print(model_1_table)
    #Run for Model 2
    model_2_table = tabulate_condition(model_2(paper_df),"summarized")'''

    #Run for Model 3
    #model_3_table_2 = tabulate_condition( model_3(paper_df_emb),"summarized")
    
    '''if os.path.exists('test_pdfs'):
        # 获取文件夹内的所有文件和子文件夹
        entries = os.listdir('test_pdfs')
        # 过滤出所有文件，排除子文件夹
        files = [f for f in entries if os.path.isfile(os.path.join('test_pdfs', f))]
        print(files)'''
    '''load_paper('test10.csv')
    paper_df_emb = add_emb(load_paper('test10.csv'))
    #store embedding to csv
    df_to_csv(paper_df_emb, 'test10_emb.csv')
    '''
    
    paper_df=load_paper("test10.csv")
    model_2_table = tabulate_condition(model_2(paper_df),"summarized")
    df_to_csv(model_2_table, 'test10_model2.csv')
    
