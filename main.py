from dotenv import load_dotenv
from pydantic import BaseModel

from vision_agents.plugins import getstream, gemini
from vision_agents.core import Agent, AgentLauncher, Runner, User

load_dotenv()

agent_configs = {}

class JoinRequestData(BaseModel):
    id: str
    name: str
    image: str
    call_id: str
    call_type: str
    instructions: str

async def create_agent(**kwargs) -> Agent:
    return Agent(
        agent_user = User(
            id = "temp_id",
            name = "temp_name"
        ),
        llm = gemini.Realtime(),
        edge = getstream.Edge()
    )

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs):
    await agent.close()
    config = agent_configs.pop(call_id)

    mainAgent = Agent(
        agent_user = User(
            id = config["id"],
            name = config["name"],
            image = config["image"]
        ),
        llm = gemini.Realtime(),
        edge = getstream.Edge(),
        instructions = config["instructions"]
    )

    call = await mainAgent.create_call(call_type, call_id)

    async with mainAgent.join(call):
        await mainAgent.simple_response(
            """
                The response language to the user should be determined by the instructions. Respond in the same language as specified in that section.
                When joining a meeting, you need to greet the user, introduce yourself, and state the objective of the meeting.
                The response style must be concise and direct. Do not provide tangential information unrelated to the question.
                The information provided must be clearly verified and not fabricated. All information must come from reliable and official sources on the internet.
            """
        )
        await mainAgent.finish()

runner = Runner(
    AgentLauncher(
        join_call = join_call,
        max_sessions_per_call = 1,
        create_agent = create_agent
    )
)

@runner.fast_api.post("/api/stream/agent/join")
async def join(data: JoinRequestData):
    call_id = data.call_id

    agent_configs[call_id] = {
        "image": data.image,
        "name": data.name,
        "id": data.id,
        "instructions": data.instructions
    }

    await runner._launcher.start_session(call_id = call_id, call_type = data.call_type)
    return { "message": "Agent đang được kết nối với cuộc họp . . ." }

if __name__ == "__main__":
    runner.cli()