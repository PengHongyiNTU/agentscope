from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
import dotenv
import os
from agentscope.formatter import DashScopeChatFormatter
from agentscope.evaluate import PersonaMemBenchmark
from typing import Literal
from pydantic import BaseModel

if __name__ == "__main__":

    async def create_agent():
        dotenv.load_dotenv()
        SYS_PROMPT = """You are a helpful assistant.
        You will be given a question from a user along with several options.
        Your task is to choose the most appropriate response from the options
        and give your final answer (a), (b), (c), or (d). 
        Directly output the label of the chosen option without any additional explanation.
        """
        agent = ReActAgent(
            name="jarvis",
            sys_prompt=SYS_PROMPT,
            model=DashScopeChatModel(
                model_name="qwen-max",
                api_key=os.environ["DASHSCOPE_API_KEY"],
                stream=True,
                enable_thinking=False,
            ),
            formatter=DashScopeChatFormatter(),
        )

        benchmark = PersonaMemBenchmark(
            data_dir="./data/personamem",
            split="32k",
        )

        data = benchmark.dataset[0]
        question = data["user_question_or_message"]
        all_options = data["all_options"]
        instructions = """Find the most appropriate model response
        and give your final answer (a), (b), (c), or (d)"""

        class MCQAnswer(BaseModel):
            answers: Literal["(a)", "(b)", "(c)", "(d)"]

        content = f"""Here is a question from a user:{question}
        Here are the options: {all_options} \n
        {instructions}
        """

        msg = Msg(
            name="user",
            content=content,
            role="user",
        )
        

        await agent(msg, structured_model=MCQAnswer)

    import asyncio

    asyncio.run(create_agent())
