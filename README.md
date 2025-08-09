# AI Health Assist

AI Health Assist is a Streamlit-based web application that helps users triage symptoms and provides health information links. Enter your symptoms in plain English, and the app will analyze urgency and suggest relevant resources.

## Features
- Symptom triage with urgency levels (Urgent, Soon, Routine)
- Explanations for triage results
- Direct links to trusted health resources for common symptoms
- Easy-to-use Streamlit interface

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/rayyansyed110/ai-health-assist.git
   cd ai-health-assist
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # On Windows
   # Or
   source .venv/bin/activate    # On macOS/Linux
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the App
```sh
python -m streamlit run app.py
```

## Deployment
You can deploy this app to [Streamlit Cloud](https://share.streamlit.io/). Push your changes to GitHub and connect your repo in Streamlit Cloud.

## Optional: Enable AI Symptom Boost
Add your Hugging Face token as `HF_TOKEN` in Streamlit Cloud → Settings → Secrets for enhanced AI features.

## License
MIT
