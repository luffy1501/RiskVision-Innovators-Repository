import openai

def setup_openai(api_key):
    openai.api_key = api_key

def get_mitigation_strategy(risk_type, risk_details):
    prompt = f"Provide a mitigation strategy for {risk_type} risk given the following details: {risk_details}"
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=150)
    return response.choices[0].text.strip()
