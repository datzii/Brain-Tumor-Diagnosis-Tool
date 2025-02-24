import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Define a tool that searches the web for information.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."

def create_agent():
    model_client = OpenAIChatCompletionClient(
        model="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
        api_key="placeholder",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )
    global agent

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="You are a helpful assitant agent. Use tools if you need it to solve tasks.",
        
    )


async def execute_assitant_query(input = "Find information on AutoGen") -> None:
    global agent

    create_agent()

    response = await agent.on_messages(
        [TextMessage(content=input, source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.inner_messages)
    print(f"-- result {response.chat_message}")



asyncio.run(execute_assitant_query())  # Run the async function
