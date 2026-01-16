# ManavaAI - Complete Project

This directory contains the complete, finalized version of the ManavaAI project, consisting of a Python FastAPI backend and a Node.js/Vite frontend.

## 📁 Structure

- **backend/**: Contains the FastAPI application, API endpoints, and detection logic.
- **frontend/**: Contains the web interface built with HTML/JS (or framework if applicable).

## 🚀 Getting Started

### 1. Backend Setup

Prerequisites: Python 3.8+

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You might need to install `torch` separately depending on your system, or if it's not in requirements.*

3.  Run the API server:
    ```bash
    python verify_api.py
    ```
    Or if using uvicorn directly:
    ```bash
    uvicorn app.main:app --reload
    ```
    The backend usually runs on `http://localhost:8000`.

### 2. Frontend Setup

Prerequisites: Node.js & npm

1.  Navigate to the `frontend` directory:
    ```bash
    cd ../frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend will be available at the URL shown in the terminal (usually `http://localhost:5173` or similar).

## 🔗 Integration

Ensure the frontend is configured to call the backend API at the correct URL (default `http://localhost:8000`). Check `frontend/script.js` or configuration files if connection issues arise.
