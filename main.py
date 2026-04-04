import deepeval
import os
from dotenv import load_dotenv
from openai import api_key

load_dotenv()

DEEP_EVAL_API_KEY = os.getenv("DEEV_EVAL_KEY")



def main():
    print("Hello from test-ai-llm-apps!")
    print(DEEP_EVAL_API_KEY)


if __name__ == "__main__":
    main()
