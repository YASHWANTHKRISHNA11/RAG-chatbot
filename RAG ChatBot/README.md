# RAG AI Assistant

A powerful Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Pinecone, and Groq. Upload your documents, and ask questions to get intelligent, context-aware answers powered by advanced language models.

## Features

- 📄 **Multi-format Document Support**: Upload PDF, Word (DOCX), CSV, Excel (XLSX/XLS), TXT, Markdown, and other text files
- 🚀 **Fast Embeddings**: Uses Hugging Face's sentence-transformers for efficient document encoding
- 🔍 **Vector Search**: Leverages Pinecone for scalable semantic search
- 🤖 **Multiple LLM Models**: Support for various Groq models for flexible AI responses
- 💾 **Namespace Organization**: Organize different document sets with isolated namespaces
- 🎨 **Professional UI**: Beautiful Streamlit interface with premium styling
- ⚡ **Real-time Processing**: Stream responses and get instant answers

## Architecture

This project uses the RAG pattern:
1. **Ingestion**: Documents are loaded and split into chunks
2. **Embedding**: Text chunks are converted to vector embeddings
3. **Storage**: Embeddings are stored in Pinecone vector database
4. **Retrieval**: User queries retrieve relevant document chunks
5. **Generation**: Retrieved context is passed to Groq LLM for answer generation

## Prerequisites

- Python 3.8+
- API keys for:
  - **Pinecone**: Vector database service
  - **Groq**: LLM provider
  - **Hugging Face** (optional): For embeddings (default model is free)

## Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create a `.env` file** in the project root with your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_pinecone_index_name
   GROQ_API_KEY=your_groq_api_key
   ```

2. **Get your API keys**:
   - **Pinecone**: Sign up at [pinecone.io](https://www.pinecone.io/)
   - **Groq**: Sign up at [console.groq.com](https://console.groq.com/)
   - **Hugging Face**: Default embeddings are free; sign up at [huggingface.co](https://huggingface.co/) if needed

## Usage

### Start the Application

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### Upload Documents

1. Use the sidebar to upload documents (PDF, DOCX, CSV, XLSX, TXT, etc.)
2. Specify a namespace to organize your document collection
3. The documents will be processed, embedded, and stored in Pinecone

### Ask Questions

1. Type your question in the chat interface
2. Select which namespace to search
3. Choose your preferred Groq model
4. Get intelligent answers based on your documents

## Supported File Types

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Adobe PDF documents |
| Word | `.docx` | Microsoft Word documents |
| CSV | `.csv` | Comma-separated values |
| Excel | `.xlsx`, `.xls` | Microsoft Excel spreadsheets |
| Text | `.txt` | Plain text files |
| Markdown | `.md` | Markdown formatted files |
| Others | `.log`, `.config`, etc. | Text-based files |

## Project Structure

```
RAG ChatBot/
├── streamlit_app.py       # Main Streamlit application UI
├── rag_chain.py           # RAG chain implementation with LangChain
├── ingest.py              # Document ingestion and processing
├── config.py              # Configuration and environment variables
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
├── runtime.txt            # Python version specification
├── temp_uploads/          # Temporary storage for uploaded files
└── README.md              # This file
```

## Key Components

### `streamlit_app.py`
Main web interface handling:
- Document upload
- Chat interface
- Model selection
- Namespace management
- Custom styling

### `rag_chain.py`
RAG implementation including:
- Embedding model setup
- Pinecone vector store initialization
- Retrieval chain configuration
- Groq LLM integration

### `ingest.py`
Document processing pipeline:
- Multi-format document loading
- Text splitting and chunking
- Document metadata handling
- Vector embedding storage

### `config.py`
Environment configuration:
- API key management
- Credential loading from `.env`

## Dependencies

- **Streamlit**: Web UI framework
- **LangChain**: RAG framework and LLM orchestration
- **Pinecone**: Vector database
- **Groq**: Language model inference
- **Hugging Face**: Embeddings and transformers
- **PyPDF**: PDF processing
- **python-dotenv**: Environment variable management
- **numpy, sentence-transformers**: ML utilities

See `requirements.txt` for complete list with versions.

## Environment Variables

```bash
PINECONE_API_KEY     # Your Pinecone API key
PINECONE_INDEX       # Your Pinecone index name
GROQ_API_KEY         # Your Groq API key
```

## Troubleshooting

### API Key Errors
- Verify all keys are correctly set in `.env`
- Restart the Streamlit app after updating `.env`
- Check API key validity in respective dashboards

### Document Upload Issues
- Ensure file format is supported
- Check file size limits
- Verify temp_uploads directory permissions

### Slow Performance
- Reduce document chunk size in `ingest.py`
- Use smaller retrieval k value in `rag_chain.py`
- Consider using a faster Groq model

### Pinecone Connection Issues
- Verify PINECONE_INDEX name matches your index
- Check API key has correct permissions
- Ensure Pinecone account quota isn't exceeded

## Advanced Usage

### Customize Embedding Model
Edit `rag_chain.py`:
```python
_embedding_model = HuggingFaceEmbeddings(
    model_name="your-model-name"
)
```

### Adjust Retrieval Parameters
Modify in `rag_chain.py`:
```python
search_kwargs={"k": 4}  # Number of retrieved documents
```

### Change Prompt Template
Update the template in `rag_chain.py` for custom behavior

## Performance Tips

1. **Use appropriate chunk sizes**: Larger chunks = faster processing, less granular matching
2. **Optimize k value**: More retrieved docs = better context, slower responses
3. **Choose suitable Groq models**: Smaller models are faster, larger are more intelligent
4. **Batch document uploads**: Process multiple files together for efficiency

## License

This project is open source. Modify and distribute as needed.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API provider documentation
3. Check LangChain and Pinecone documentation

## Future Enhancements

- [ ] Chat history and conversation management
- [ ] User authentication
- [ ] Document deletion and management UI
- [ ] Advanced analytics and usage metrics
- [ ] Hybrid search (keyword + semantic)
- [ ] Multi-modal document support
- [ ] Export conversation features
- [ ] API endpoint deployment

---

**Built with LangChain, Pinecone, Groq, and Streamlit**
