# Fetal health FastAPI Application


## Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Docker (optional, if you want to use Docker)

## Getting Started

### 1. Clone the repository

```bash
git clone [<repository-url>](https://github.com/DimitriKwihangana/MediStat)
cd MEDISTAT
```

### 2. Set up a virtual environment (recommended)

Create and activate a virtual environment:

```bash
# On Linux/MacOS
python3 -m venv env
source env/bin/activate

# On Windows
python -m venv env
env\Scripts\activate
```

### 3. Install dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 4. Run the application

Start the FastAPI application using the following command:

```bash
uvicorn main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

### 5. Access the API documentation

FastAPI automatically provides interactive API documentation. Visit the following endpoints:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Optional: Using Docker

If you prefer running the application in Docker:

### 1. Build the Docker image

```bash
docker build -t medistat-app .
```

### 2. Run the Docker container

```bash
docker run -p 8000:8000 medistat-app
```

## Additional Information


- For load testing, use `locustfile.py` to simulate traffic.

### Locust file for testing apis
  ![locust](https://github.com/user-attachments/assets/4fe189da-355f-4bdc-b487-71be978368ce)
## Front-end Link

## YouTube Link for demonstration

