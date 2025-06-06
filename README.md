# News Research Tool

## Overview
The **News Research Tool** is a Python-based application that allows users to process news article URLs, extract and split their content, create embeddings using OpenAI, and perform question-answering on the processed data. The tool uses `Streamlit` for the user interface and integrates with OpenAI's GPT models for natural language processing.

## Features
- Input up to three news article URLs for processing.
- Extract and split content into manageable chunks.
- Generate embeddings using OpenAI's `OpenAIEmbeddings`.
- Store embeddings locally using `FAISS` for efficient retrieval.
- Perform question-answering on the processed data using OpenAI's GPT models.
- User-friendly interface powered by `Streamlit`.

## Requirements
- Python 3.12 or higher
- Required Python packages (listed in `requirements.txt`):
  - `langchain==0.0.304`
  - `python-dotenv==1.0.0`
  - `streamlit==1.26.0`
  - `unstructured==0.9.2`
  - `libmagic==1.0`
  - `python-magic==0.4.27`
  - `openai==0.28.0`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the `.env` file with your OpenAI API key:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run main.py
   ```

2. Use the sidebar to input up to three news article URLs.

3. Click the **Process URLs** button to process the articles. The tool will:
   - Extract content from the URLs.
   - Split the content into chunks.
   - Generate embeddings and save them locally in a pickle file.

4. Enter a question in the input box to query the processed data.

5. View the answer generated by the model in the main interface.

## File Structure
- `main.py`: Main application script.
- `.env`: Environment variables file (contains the OpenAI API key).
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Files and directories to ignore in version control.

## Troubleshooting
- Ensure the OpenAI API key in the `.env` file is valid.
- If the embeddings file (`faiss_store_embedding.pkl`) is missing, reprocess the URLs.
- Update the `langchain` library if you encounter compatibility issues.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
