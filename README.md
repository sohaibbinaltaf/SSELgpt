🔬 SSEL-GPT

SSEL-GPT is a lightweight Retrieval-Augmented Generation (RAG) application for asking questions about the Smart Systems Engineering Lab (SSEL) Activity Report 2025.

It uses:

Python

Streamlit for the UI

FAISS for vector similarity search

Sentence Transformers for local embeddings

Groq API for fast LLM generation

PyPDF for PDF extraction

Google Drive as the source location of the report

The application is designed to run on Google Colab and Streamlit Community Cloud with only three files.

1. Project structure

SSEL-GPT/
├── app.py
├── requirements.txt
└── README.md

No separate .env, database, vector-store directory, or LangChain project is required.

2. How the RAG pipeline works

On application startup, SSEL-GPT:

Downloads the SSEL Activity Report PDF from the configured Google Drive URL.

Extracts text page-by-page using pypdf.

Splits the report into smaller, page-aware chunks and carries section headings into retrieval.

Generates embeddings using the free/open all-MiniLM-L6-v2 Sentence Transformer.

Builds both a FAISS dense-vector index and a dependency-free BM25 lexical index.

Expands common SSEL terminology such as IoT, UGV, UAV, and SUMO Robot before searching.

Combines semantic and exact-term retrieval so names, acronyms, dates, headings, and project titles are easier to find.

Retrieves diverse evidence from the most relevant report pages; there is no brittle fixed similarity cutoff.

Sends the retrieved evidence plus the question to the selected Groq model.

Generates a grounded response with report-page citations such as [Page 20].

Optionally displays the retrieved chunks and similarity scores.

The FAISS index and embedding model are cached with Streamlit's st.cache_resource, so the expensive indexing operation is normally performed once per application process.

3. Source document

The default source is the SSEL Activity Report 2025:

https://drive.google.com/file/d/1qhGYVTtZ2IZYYmNhi3Lvhe5x7qmoNzj9/view?usp=drive_link

The report is expected to be accessible to the deployment environment.

The application is intentionally report-grounded. If the retrieved report evidence does not support an answer, SSEL-GPT is instructed to say so rather than inventing information.

4. Groq API key

SSEL-GPT requires a Groq API key for answer generation.

Local / Colab

Set an environment variable:

import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

or paste the key into the application's sidebar.

Streamlit Community Cloud

Add the following under Settings → Secrets:

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

Do not commit the API key to GitHub.

5. Install and run locally

pip install -r requirements.txt
streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

6. Run in Google Colab

Upload app.py and requirements.txt to Colab, then run:

!pip install -r requirements.txt

Start Streamlit:

!streamlit run app.py &>/content/streamlit.log &

For a temporary public URL, you can use a tunneling service such as Cloudflare Tunnel or another service available in your Colab environment.

Example using localtunnel:

!npm install -g localtunnel
!lt --port 8501

Keep the Colab runtime active while using the application.

7. Deploy to Streamlit Community Cloud

Create a GitHub repository.

Put these three files in the repository root:

app.py

requirements.txt

README.md

Open Streamlit Community Cloud.

Create a new app.

Select the repository and app.py.

Add the Groq API key under the application's Secrets.

Deploy.

The app downloads and indexes the report automatically after deployment.

Important deployment note

The first startup can take longer because the application needs to:

download the PDF,

download the embedding model,

extract 143 pages,

generate embeddings,

construct the FAISS index.

After st.cache_resource is populated, subsequent interactions do not repeat the indexing work during the same app process.

8. Application controls

SSEL-GPT includes several useful controls in the sidebar.

LLM model

The app provides selectable Groq models, including:

openai/gpt-oss-120b
openai/gpt-oss-20b
qwen/qwen3.6-27b

Model availability and lifecycle can change on Groq, so the list is intentionally kept in app.py.

Response size

Four options are provided:

Concise — essential answer

Standard — normal answer

Detailed — more explanation and report details

Comprehensive — extensive answer based on retrieved evidence

Temperature

Controls generation variability.

Recommended values:

0.0–0.2  → highly factual / deterministic
0.3–0.5  → balanced
0.6–1.0  → more creative

For an institutional report, 0.0–0.3 is recommended.

Evidence chunks

Controls how many hybrid-search results are passed to the LLM. The default is 7.
For lists, member directories, projects, and timelines, 7–10 is usually useful.

Retrieval mode

The app uses hybrid retrieval rather than a single vector similarity threshold:

Balanced — recommended default; combines semantic and lexical matching.

Precise — favors exact report terminology, names, headings, and phrases.

Broad — favors semantic similarity for conceptual questions.

The previous fixed similarity cutoff has been removed because it could reject valid answers such as
questions about the IoT workshop, UGV projects, or the Members section.

Show retrieved sources

Displays the retrieved report pages, similarity scores, and short previews.

Use conversation context

Allows follow-up questions such as:

Who is the PI?

followed by:

What is the budget?

The recent conversation is incorporated into retrieval and generation.

Rebuild index

The Rebuild index button clears the Streamlit resource cache and downloads/reprocesses the report again.

Use this if the source PDF is updated.

9. Example questions

SSEL-GPT is designed to answer questions such as:

What are the four main research themes of SSEL?

Who is the leader of SSEL?

Who are the postdoctoral researchers?

What projects are led by Dr. Sohaib Bin Altaf Khattak?

What is the budget of the smart indoor positioning project?

Which SSEL projects are completed?

What training programs were conducted?

Which workshops did Dr. Sohaib deliver?

What seminars were organized in 2025?

What awards did SSEL members receive?

How many Q1 and Q2 journal publications are reported?

What are the main research outcomes?

What software platforms were developed?

What is the SSEL infrastructure?

What are the next-year action plan objectives?

Compare SSEL's research themes with its research projects.

Give me a timeline of major SSEL activities in 2025.

Summarize SSEL's 2025 achievements for a presentation.

Which projects involve IoT and smart cities?

Which projects involve AI?

Which activities involved international collaboration?

It can also handle follow-up questions using conversation context.

10. Why FAISS + local embeddings?

This architecture keeps the retrieval layer inexpensive and simple.

SSEL PDF
   ↓
PyPDF text extraction
   ↓
Page-aware chunking
   ↓
Sentence Transformer embeddings
   ↓
FAISS vector index
   ↓
User question
   ↓
Question embedding
   ↓
Top-K similarity search
   ↓
Retrieved report evidence
   ↓
Groq LLM
   ↓
Grounded answer

No paid vector database is required.

11. Grounding and hallucination control

The system prompt tells the LLM to:

use the retrieved report evidence,

avoid inventing facts,

preserve names, dates, budgets, project statuses, and figures,

acknowledge when the report does not establish an answer,

cite report pages where useful,

stay focused on SSEL.

This makes the application suitable for an institutional activity-report knowledge assistant.

However, RAG does not mathematically guarantee zero hallucinations. For high-stakes institutional use, retrieved evidence should still be reviewed.

12. Updating the report

If a new SSEL report replaces the current report:

Change GOOGLE_DRIVE_URL in app.py.

Change PDF_NAME if desired.

Redeploy or press Rebuild index.

The rest of the RAG pipeline does not need to change.

13. Important notes

Google Drive permissions

The PDF must be accessible to the application. If the Drive file requires an interactive login that gdown cannot complete, download the PDF manually and replace the download mechanism in app.py.

Embedding model

The application uses:

sentence-transformers/all-MiniLM-L6-v2

This is intentionally lightweight and suitable for a free deployment.

FAISS

The application uses:

faiss.IndexFlatIP

with normalized embeddings. This makes the inner product equivalent to cosine similarity.

Privacy

Do not put confidential reports or confidential API keys into a public repository.

The report is sent only as retrieved text to the selected Groq model for answer generation.

14. Customization ideas

The current three-file implementation intentionally keeps the project simple. Future versions could add:

multiple SSEL reports,

year filtering,

project/person filters,

cross-encoder reranking,

document upload,

downloadable answers,

chat export,

analytics,

authentication,

administrator-only knowledge-base updates,

multilingual Arabic/English answers,

publication-specific retrieval,

automatic report comparison across years.

For the current requirement, these are intentionally not added so the application remains easy to deploy and maintain.

15. Recommended default configuration

For factual questions about the SSEL report:

Model: openai/gpt-oss-120b
Response size: Standard
Temperature: 0.1
Retrieval mode: Balanced
Evidence chunks: 7
Show sources: Enabled
Conversation context: Enabled

16. License / usage

This implementation is provided as a project template for the SSEL-GPT application. Check the applicable licenses and institutional permissions for the report, embedding model, FAISS, Streamlit, and Groq services before public deployment.
