from typing import Literal
from agentscope.message import Msg
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
import dotenv
import os 
from pydantic import BaseModel
import asyncio


class DesiredResponse(BaseModel):
    rating: Literal["excellent", "good", "fair", "poor"]


async def create_agent():
    dotenv.load_dotenv()
    agent = ReActAgent(
        name="test-agent",
        sys_prompt="""You are a helpful assistant""",
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=True,  # Streaming enabled
            enable_thinking=False),
        formatter=DashScopeChatFormatter()
    )

    msg = Msg(
        name="user",
        content="Hello, how can you assist me today?",
        role="user",
    )
    
    print("Sending message to agent...")
    
    # Call agent with structured output model
    response = await agent(msg, structured_model=DesiredResponse)
    
    # View all aspects of the response
    print("\n=== AGENT RESPONSE ANALYSIS ===")
    print(f"Agent name: {response.name}")
    print(f"Role: {response.role}")
    print(f"Content: {response.content}")
    print(f"Metadata: {response.metadata}")
    
    # Access structured output if available
    if response.metadata:
        print(f"\nStructured Response: {response.metadata.get('response', 'N/A')}")
        print(f"Rating: {response.metadata.get('rating', 'N/A')}")
    
    return response


if __name__ == "__main__":
    # Optional: Enable detailed logging
    # setup_logger(level=logging.INFO)
    
    print("Running agent...")
    res = asyncio.run(create_agent())
    print(f"\nFinal result: {res}")