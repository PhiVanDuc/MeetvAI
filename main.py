import contextvars
from dotenv import load_dotenv
from pydantic import BaseModel

from vision_agents.plugins import getstream, gemini
from vision_agents.core import Agent, AgentLauncher, Runner, User

load_dotenv()

current_call_config = contextvars.ContextVar("current_call_config", default = None)

class JoinRequestData(BaseModel):
    id: str
    name: str
    image: str
    call_id: str
    call_type: str
    instructions: str

async def create_agent(**kwargs) -> Agent:
    config = current_call_config.get()

    if config is None:
        image = ""
        id = "warmup_id"
        name = "Warmup Agent"
        instructions = "You are a helpful assistant."
    else:
        id = config["id"]
        name = config["name"]
        image = config["image"]
        
        instructions = f"""
            {config["instructions"]}

            LANGUAGE PROTOCOL:
            1. Initially, speak in the same language as these instructions.
            2. Maintain this language unless the user explicitly asks to switch.
            3. If a switch is requested, use the new language for all subsequent interactions.

            RESPONSE RULES:
            1. THIS IS IMPORTANTEST RULE - Never use markdown, bold, italic, bullet points, numbered lists, or any text formatting. Only use formatting that is compatible with text-to-speech engines.
            2. Be concise and direct. Do not provide tangential information.
            3. All information must be verified and from reliable sources.
        """

    llm = gemini.Realtime(model = "gemini-3.1-flash-live-preview")

    return Agent(
        llm = llm,
        agent_user = User(
            id = id,
            name = name,
            image = image
        ),
        edge = getstream.Edge(),
        instructions = instructions
    )

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs):
    await agent.authenticate()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()

launcher = AgentLauncher(
    join_call = join_call,
    max_sessions_per_call = 1,
    create_agent = create_agent
)

runner = Runner(launcher)

@runner.fast_api.post("/api/stream/agent/join")
async def join(data: JoinRequestData):
    token = current_call_config.set({
        "id": data.id,
        "name": data.name,
        "image": data.image,
        "instructions": data.instructions
    })

    try:
        await launcher.start_session(call_id = data.call_id, call_type = data.call_type)
    finally:
        current_call_config.reset(token)

if __name__ == "__main__":
    runner.cli()

# from dotenv import load_dotenv

# from vision_agents.core import Agent, AgentLauncher, User, Runner
# from vision_agents.plugins import getstream, gemini

# load_dotenv()


# async def create_agent(**kwargs) -> Agent:
#     return Agent(
#         edge=getstream.Edge(),
#         agent_user=User(name="Assistant", id="agent"),
#         instructions="You're a helpful voice assistant. Be concise.",
#         llm=gemini.Realtime(),
#     )


# async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
#     call = await agent.create_call(call_type, call_id)
#     async with agent.join(call):
#         await agent.simple_response("Greet the user")
#         await agent.finish()


# if __name__ == "__main__":
#     Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()