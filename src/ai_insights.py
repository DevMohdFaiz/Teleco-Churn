from groq import Groq
import os
import streamlit as st
from dotenv import get_key
from typing import Dict


def get_api_key():
    """Get Groq API Key from environment"""
    GROQ_API_KEY = get_key(".env", "GROQ_API_KEY")
    if GROQ_API_KEY is None:
        try: 
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        except:
            GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    return GROQ_API_KEY

GROQ_API_KEY = get_api_key()

def call_groq(GROQ_API_KEY=GROQ_API_KEY):
    if Groq is None:
        return None
    if not GROQ_API_KEY:
        return f"GROQ API KEY not found!\n Pls check your environment variables"
    
    return Groq(api_key=GROQ_API_KEY)


def build_context(churn_pred, churn_probability, top_churn_factors, data_features):
    """
    Build the prompt context to be fed to the LLM
    Args:
        churn_pred (int): churn_prediction
        churn_probabaility (float): churn_probability
        top_churn_factors (np.array): top features influencing churn prediction from SHAP
        data_features (list): names of features fed to SHAP
    Returns:
        prompt: prompt to be fed to LLM
    """
    prompt = f"""You are a customer retention analyst for a telecommunications company. Analyze this churn prediction:

    CUSTOMER CHURN RISK: {churn_probability}%
    PREDICTION: {"WILL CHURN" if churn_pred == 1 else "WILL STAY"}

    TOP FACTORS DRIVING PREDICTION:
    {top_churn_factors}

    KEY CUSTOMER ATTRIBUTES:
    {data_features}

    CONTEXT:
    - `increase_pct`: Percentage increase in monthly charges vs. customer's historical average
    - `tenure`: Months as a customer
    - `customer_segment`: Behavioral cluster 

    Common telecom churn drivers include: contract lenght (Monthly, Yearly or Biennial),
    payment method, number of services subscribed to, price increase, multiple lines, customer segment 
    and whether the customer is a senior citizen.

    Provide analysis in this format:

    RISK ASSESSMENT (2-3) sentences:
    [Explain why this customer might leave, considering telecom industry context]

    RETENTION STRATEGY:
    - [Action 1 - address primary risk factor]
    - [Action 2 - leverage protective factors]
    - [Action 3 - proactive engagement]

    Keep it actionable and insightful for customer service teams and retention managers."""
    return prompt


def groq_chat(prompt: Dict, model:str= "openai/gpt-oss-120b", max_tokens: int=2048, num_retries:int =3)-> Dict[str, str]:
    """
    Make a call to the Groq API with the query and context

    Args:
        mesages (dict): query with full context
        model (str): AI model to use
        max_tokens (int): maximum number of tokens

    Returns:
        messages (dict): Dictionary with response and metadata
    
    Raises:
        Exception if API call fails
    """
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(1, num_retries+1):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_tokens,
                temperature=0.4
            )   
            response_text = chat_completion.choices[0].message.content        
            return {
                'provider': 'groq',
                'model': model,
                'response': response_text,
                "usage": {
                    "prompt_tokens": chat_completion.usage.prompt_tokens,
                    "completion_tokens": chat_completion.usage.completion_tokens,
                    "total_tokens": chat_completion.usage.total_tokens
                    }
                }
        except Exception as e:
            if attempt == num_retries-1:
                raise RuntimeError(f"Groq API call failed after {num_retries} attempts: {str(e)}")
            continue

    raise RuntimeError(f"Unexpected error in Groq API")
