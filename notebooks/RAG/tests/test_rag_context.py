
# 1. Get the path to the project root relative to this file
import os
import sys

# Add parent directories to path for module resolution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from langchain.messages import HumanMessage

from modules.langchain_agent import agent

rag_agent = agent()
# response = rag_agent.invoke({
#     "messages": [HumanMessage(content="What is he total experience of Gautam?")]
# })
# print(response["messages"][-1].content)


from deepeval.test_case import LLMTestCase, llm_test_case
from deepeval.dataset import EvaluationDataset

llmTest = LLMTestCase(
    input="What is Gautam skill set?"
    actual_output=rag_agent.invoke({
    "messages": [HumanMessage(content="What is he total experience of Gautam?")]
})
    expected_output="Java, Python, Langchain"
)

data_set = EvaluationDataset()
data_set.add_test_case(llmTest)

