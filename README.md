# Data Management Final Project 

## Introduction 
This project provides a backend API for ranking data using a **FastAPI** server. The API allows sorting data based on various ranking methods and retrieving ranking stability. 

Additionally, this project features a **frontend application built with React**. The frontend serves as an interactive interface for users to select attributes, compute stable rankings using different algorithms, and display results in a table format. It also offers insights by comparing the different rankings that were returned. The application is designed to enhance user experience by allowing for easy interaction with the ranking data derived from the backend API.

## Algorithms
The system utilizes two ranking algorithms **Raysweeping** and **Randomized Rounding** from the paper *"On Obtaining Stable Ranking"* (Asudeh et al., PVLDB 2018). Each algorithm is designed to compute rankings based on user-defined constraints.

1. **Ray Sweeping Algorithm**

    The Ray Sweeping algorithm processes rankings based on an interest region defined by the user's constraints. The key functions involved in this algorithm are:
    - `compute_interest_region`: Given a list of constraints, computes the valid range of angles that define the interest region.
    - `compute_raysweeping_heap`: Creates a heap of ranking regions based on the initial ranking determined by the minimum angle in the interest region. It computes the stability of rankings based on the ordering exchange angles between adjacent players in the rankings. The stability is calculated as the ratio of the angle range to the total range.
    - `process_raysweeping`: Extracts the stability and angle from the heap, and computes the ranking function. It then finds the actual ranking from the data, selecting the top results and formatting the output for return.

2. **Randomized Rounding Algorithm**

    The Randomized Rounding algorithm generates rankings by sampling weight vectors within the defined interest region. The key functions involved in this algorithm are:
    - `generate_randomized_angles`: Generates a specified number of random angles (each of which corresponds to a weight vector) uniformly distributed within the interest region's bounds.
    - `randomized_get_next`: Performs the core sampling process. It samples weight vectors from the interest region and computes the rankings for these samples. Each ranking is assigned a unique ID, and the function counts how often each ranking appears. It then selects the most frequently occurring rankings along with their associated stability scores.
    - `process_randomized_rounding`: Utilizes the sampled rankings to compute actual rankings, selecting the top results based on user specifications and formatting them for output.

## Dataset 

The <a href="https://www.kaggle.com/datasets/crxxom/chess-gm-players/data">Chess All GM players Statistics 2023</a> dataset contains statistics of all Grandmaster (GM) titled players on Chess.com as of July 17, 2023. It includes details like usernames, ratings (FIDE, rapid, blitz, bullet), game history, and other relevant data. 

## Project Structure 

```
DM_FINAL_PROJECT/
│── DM-app
│   │── src/
│       │── components/           # React components and hooks divided by logic
│           │── ...
│       │── consts/
│           │── ...
│       │── App.tsx               # Main app layout
│       │──index.tsx
│   │── index.html                # App entry point
│── Server/
│   ├── Data/                     # Directory for data files
│   ├── backend_services.py       # Core logic for ranking algorithms
│   ├── main.py                   # FastAPI server
│   ├── requirements.txt          # Dependencies for the project
│   ├── res.json                  # Example response output
│   ├── test_backend_services.py  # Unit tests for backend services
│   ├── test_server.py            # Tests for the API server
│   ├── README.txt                # Project documentation
│── .vscode/                      # VS Code settings
│── __pycache__/                  # Compiled Python files
```

## Setup Instructions 

### 1. Clone the repository
```bash
git clone <repository-url>
cd DM_FINAL_PROJECT/Server
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup frontend
Navigate to the `DM-app/` directory and install dependencies:
```bash
cd /DM_FINAL_PROJECT/DM-app
npm install
```

## Running the API Server  
Start the FastAPI server:
```bash
cd DM_FINAL_PROJECT/Server
uvicorn main:app --reload
```
The API will be accessible at:  
- **Swagger Docs:** `http://127.0.0.1:8000/docs`  
- **Redoc Docs:** `http://127.0.0.1:8000/redoc`  

## Running the App
```bash
cd /DM_FINAL_PROJECT/DM-app
npm run dev
```

## Endpoints

### 1. Get Column Names
- **Endpoint:** `GET /columns`  
- **Response:** Returns column names from the dataset.  

#### Example Usage:
```python
import requests
response = requests.get("http://127.0.0.1:8000/columns")
print(response.json())
```

### 2. Get Ranking
- **Endpoint:** `POST /ranking`  
- **Request Body:** JSON with constraints, ranking method, and column selection.  
- **Response:** Returns ranked data with ranking function and stability.  

#### Example Request Using Ray Sweeping Method:
```python
import requests

data = {
    "constraints": [[1, 2, "<="], [1, 1, ">="]],
    "method": "Ray Sweeping",
    "columns": ["rapid_win", "bullet_win"],
    "num_ret_tuples": 1,
    "num_of_rankings": 1
}

response = requests.post("http://127.0.0.1:8000/ranking", json=data)
print(response.json())
```

#### Example Request Using Randomized Rounding Method:
```python
import requests

data = {
    "constraints": [[1, 2, "<="], [1, 1, ">="]],
    "method": "Randomized Rounding",
    "columns": ["rapid_win", "bullet_win"],
    "num_ret_tuples": 3,
    "num_of_rankings": 2,
    "num_of_samples": 1000
}

response = requests.post("http://127.0.0.1:8000/ranking", json=data)
print(response.json())
```

#### Example Response:
```json
[
    {
        "Ranking": [
            {"user_id": 15448422, "name": "Hikaru Nakamura", "Rank": 0.5756, "followers": 1.0, "bullet_win": 0.1853},
            {"user_id": 2406471, "name": "Rogelio Jr Antonio", "Rank": 0.5217, "followers": 0.0018, "bullet_win": 1.0}
        ],
        "Ranking_Function": {"w1": 0.4791, "w2": 0.5208},
        "Stability": 0.1881
    }
]
```

### 3. Get Stability Score
- **Endpoint:** `POST /stability`  
- **Request Body:** JSON with column names and weight parameters `W1` and `W2`.
- **Response:** Returns the stability score of the ranking.

#### Example Request:
```python
import requests

data = {
    "columns": ["rapid_win", "bullet_win"],
    "W1": 0.2,  # Weight for the first ranking factor
    "W2": 0.8   # Weight for the second ranking factor
}

response = requests.post("http://127.0.0.1:8000/stability", json=data)
print(response.json())
```

## Contributors
- **Emanuel Goldman**  
- **Dana Goldberg**

