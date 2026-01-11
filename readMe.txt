1. preprocessing_phase 1

- Loads the files training_matrix, MOCA, UPSIT, RBD
- Target lebel is created
- Patient's scores are tracked using event_id
- Feature engineering is done for MOCA(calculate the mean and slope to determine the changes from first and last visit), UPSIT and RBD(if the patient has taken the test or not)
- Merged based on PATNO using left join
- Missing values are handled using SimpleImputer by replacing NaN with the median of the column. Median is robust to outliers so better than mean
- Result is final_dataset.csv


2. preprocessing_phase 2

- Used for enhancing final_dataset.csv
- Raw RBD and UPSIT datasets are loaded again to extract fine grain time series info
- The columns of rbd represent individual RBD questionnaire items (e.g., vivid dreams, injuries during sleep, sleep disturbance) and each has value 0/1. Total score is calculated
- Determines RBD mean and RBD slope. Uses polynomial to fit the scores and determine slope.
- Same is done for UPSIT
- Merged the enhanced features and fill the missing values

3. model_training 3

- Loads and cleans a biomedical dataset.
- Handles data leakage, missing values, and scaling. 
- Builds multiple ML pipelines (Logistic Regression, Random Forest, SVM, XGBoost) with SMOTE for class imbalance.
- Tunes hyperparameters via RandomizedSearchCV.
- Selects the best model based on ROC-AUC.
- Saves the trained pipeline, scaler, and feature list.

4. ingest_docs

- Finds all the pdfs and loads them
- Split into smalle chunks using RecursiveCharacterTextSplitter
-  Generate embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
- Embeddings stored in vector store

5 rag_module

- Loads the prebuilt Chroma vector store (from your ingest_documents.py script).
- Retrieves relevant chunks from your embedded PDFs using a query (like patient data).
- Sends structured + retrieved info to Groq’s LLM API for summarization and reasoning.
- Returns two reports:Doctor and patient
- groq model is llama-3.1-8b-instant
- cache is used to store the reports
- payload is setup and post request is sent which returns the json message
- Retrieval happens by extracting top -k embeddings from the vector store based on the query
- RAG reports are generated and also cached to avoid repeatative API calls and enhance performance.
- call_groq() is called twice becoz 2 reports

6 streamlit happens

- SHAP used is tree SHAP. Its useful for tree based models like random forest, xgboost. It gives the feature and contribution of that feature in the form of a dataframe





