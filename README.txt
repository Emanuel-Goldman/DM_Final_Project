# **Data Management Final Project**  
This project provides a backend API for ranking data using a **FastAPI** server. The API allows sorting data based on various ranking methods and retrieving ranking stability. The system applies methods from the paper **"On Obtaining Stable Ranking"**, utilizing the **Ray Sweeping algorithm** and a **randomized Monte Carlo approach** to compute stable rankings based on constraints and scoring functions.

---

## **Project Structure**  

```
DM_FINAL_PROJECT/
│── Server/
│   ├── Data/                    # Directory for data files
│   ├── backend_services.py       # Core logic for ranking algorithms
│   ├── main.py                   # FastAPI server
│   ├── requirements.txt          # Dependencies for the project
│   ├── res.json                  # Example response output
│   ├── test_backend_services.py  # Unit tests for backend services
│   ├── test_server.py            # Tests for the API server
│   ├── README.txt                # Project documentation
│── .vscode/                      # VS Code settings
│── __pycache__/                   # Compiled Python files
```

---

## **Setup Instructions**  

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd DM_FINAL_PROJECT/Server
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Running the API Server**  

Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The API will be accessible at:  
- **Swagger Docs:** `http://127.0.0.1:8000/docs`  
- **Redoc Docs:** `http://127.0.0.1:8000/redoc`  

---

## **Endpoints**  

### **1. Get Column Names**
- **Endpoint:** `GET /columns`  
- **Response:** Returns column names from the dataset.  

### **2. Get Ranking**
- **Endpoint:** `POST /ranking`  
- **Request Body:** JSON with constraints, ranking method, and column selection.  
- **Response:** Returns ranked data with ranking function and stability.  

#### **Example Request:**
```json
{
    "constraints": [[1, 2, "<="], [1, 1, ">="]],
    "method": "Ray Sweeping",
    "columns": ["followers", "bullet_win"],
    "num_ret_tuples": 2,
    "num_of_rankings": 1
}
```

#### **Example Response:**
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

---

## **Testing**
Run unit tests for backend services and the server using:
```bash
python -m unittest test_backend_services.py
python -m unittest test_server.py
```

---

## **Contributors**  
- **Emanuel Goldman**  
- **Dana Goldberg**  

