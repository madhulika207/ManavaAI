# ManavaAI Backend 2.0

This is the FastAPI-based backend for ManavaAI.

## Prerequisites

- Python 3.8+
- Java (required for `language_tool_python`)

## Setup & Run

1.  **Navigate to the directory**:
    ```bash
    cd backend2.0
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    # Create virtual environment (optional but recommended)
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    
    # Install requirements
    python -m pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    python -m uvicorn app.main:app --reload
    ```
    The server will start at `http://127.0.0.1:8000`.

## API Documentation

Once running, access the interactive API docs at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Testing

Run the verification script to check endpoints:
```bash
python verify_api.py
```
