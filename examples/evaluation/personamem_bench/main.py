from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
import dotenv
import os
from agentscope.formatter import DashScopeChatFormatter
from agentscope.evaluate import PersonaMemBenchmark

if __name__ == "__main__":

    # async def create_agent():
    #     dotenv.load_dotenv()
    #     agent = ReActAgent(
    #         name="jarvis",
    #         sys_prompt="You are a helpful assistant.",
    #         model=DashScopeChatModel(
    #             model_name="qwen-max",
    #             api_key=os.environ["DASHSCOPE_API_KEY"],
    #             stream=True,
    #             enable_thinking=False,
    #         ),
    #         formatter=DashScopeChatFormatter(),
    #     )

    #     msg = Msg(
    #         name="user",
    #         content="Hi! Jarvis, write me a python quick sort program.",
    #         role="user",
    #     )

    #     await agent(msg)

    # import asyncio

    # asyncio.run(create_agent())
    
    benchmark = PersonaMemBenchmark(
        data_dir="./data/personamem",
        split="32k",
    )
    print(f"Loaded {len(benchmark.dataset)} records from PersonaMem dataset.")
    for i, record in enumerate(benchmark.dataset[:3]):
        print(f"Record {i+1}: {record}")

