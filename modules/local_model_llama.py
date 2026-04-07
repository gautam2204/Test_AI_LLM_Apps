from deepeval.models import GPTModel

local_judge = GPTModel(
    model="Llama-3.2-3B-Instruct-Q4_K_M",             # Name doesn't matter, but helps in logs
    base_url="http://localhost:8080/v1", 
    api_key="needed-but-ignored"      # Llama-server doesn't need a real key
 
)

