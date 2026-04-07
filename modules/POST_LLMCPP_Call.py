import requests
def get_llm_response(user_query:str):
    # 1. Setup Configuration
    url = "http://127.0.0.1:8080/v1/chat/completions"
    api_key = "your_api_key_here"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 2. Define the Payload
    data = {
        "model": "local Model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.7
    }

    # 3. Make the Call
    response = requests.post(url, headers=headers, json=data)

    # 4. Handle the Response
    if response.status_code == 200:
        result = response.json()
        print(result['choices'][0]['message']['content'])
    else:
        print(f"Error {response.status_code}: {response.text}")
        
    return result['choices'][0]['message']['content']