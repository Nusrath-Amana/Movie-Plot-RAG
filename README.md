# Movie Plot RAG


### Prerequisites

- Python 3.8+

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Nusrath-Amana/Movie-Plot-RAG.git
   cd Movie-Plot-RAG
   ```
2. **Run the Movie Plot Pipeline**  
  - Create and activate a virtual environment:  
  ```bash
  python -m venv venv
  source venv/Scripts/activate  # On Windows
  ```
  - Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

  - Add .env file:
    - Create a .env file with the following content:
  ```env
  GOOGLE_API_KEY=your_gemini_api_key
  ```

  - Run the python script:
  ```bash
  python main.py
  ```


---

## 🧪 Example Queries

- "How does the temperature in Room A change by hour of the day?"
- "Which room had the highest temperature reading last week?"
- "How does CO₂ vary by day of the week?"
- "List the rooms from hottest to coolest by daily average temperature"
