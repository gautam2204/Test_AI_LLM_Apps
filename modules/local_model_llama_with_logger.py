import json
import asyncio
import os
from deepeval.models import GPTModel

# 1. Create a storage list for our logs
judge_interactions = []

def save_to_json(data, filename="judge_traces.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# 2. Define the Interceptor
def trace_judge(model_instance):
    original_a_generate = model_instance.a_generate

    async def patched_a_generate(prompt, schema=None):
        # Capture the request
        interaction = {
            "model": model_instance.model_name,
            "prompt": prompt,
            "schema_expected": str(schema) if schema else None,
            "response": None
        }
        
        # Execute the actual call to llama-server
        # Note: GPTModel.a_generate returns (output_text, cost)
        res, cost = await original_a_generate(prompt, schema)
        
        # Capture the response
        interaction["response"] = res
        judge_interactions.append(interaction)
        
        # Auto-save after every call
        save_to_json(judge_interactions)
        
        return res, cost

    # Apply the patch
    model_instance.a_generate = patched_a_generate
    return model_instance

# 3. Setup your Judge
local_judge = GPTModel(
    model="qwen2.5-1.5b-instruct-q4_k_m.", # Ensure this matches your llama-server model name
    base_url="http://localhost:8080/v1",
    api_key="sk-dummy"
    # REMOVED: verbose_mode=True
)

# 4. Wrap it with our tracer
# local_judge = trace_judge(local_judge)

# --- Now use local_judge in your metrics as usual ---
# from deepeval.metrics import AnswerRelevancyMetric
# metric = AnswerRelevancyMetric(model=local_judge)