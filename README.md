# Risk Management System

This project is designed to improve the risk management capabilities of a bank by utilizing generative AI to predict, assess, and mitigate various types of risks.

## Setup

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv riskmanagement-env
    source riskmanagement-env/bin/activate  # On Windows, use `riskmanagement-env\Scripts\activate`
    ```

2. **Install required dependencies:**

    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3. **Place your data files in the `data/` directory:**

    - `market_data.csv`
    - `customer_transactions.csv`
    - `operational_logs.csv`

4. **Set up OpenAI API key:**

    Update the `setup_openai` function in `chatbot/chatbot.py` with your OpenAI API key.

## Running the Project

```bash
python main.py
