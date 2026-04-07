from dotenv import load_dotenv
from pydantic import BaseModel

from vision_agents.plugins import getstream, gemini
from vision_agents.core import Agent, AgentLauncher, Runner, User
from google.genai.types import EndSensitivity, RealtimeInputConfigDict, AutomaticActivityDetectionDict

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
        edge = getstream.Edge(),
        llm = gemini.Realtime(model = "gemini-3.1-flash-live-preview")
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
        edge = getstream.Edge(),
        instructions = config["instructions"],
        llm = gemini.Realtime(model = "gemini-3.1-flash-live-preview")
    )

    call = await mainAgent.create_call(call_type, call_id)

    async with mainAgent.join(call):
        await mainAgent.simple_response(
            f"""
                User Instructions: {config["instructions"]}

                Language Protocol:
                1. Initially, you must speak in the same language as the "User Instructions" provided above.
                2. Maintain this language unless the user explicitly asks you to switch to a different language.
                3. If a switch is requested, proceed with the new language for all subsequent interactions.

                Behavioral Guidelines:
                When joining, greet the user, introduce yourself, and state the objective of the meeting based on the instructions.
                The response style must be concise and direct. Do not provide tangential information.
                All information must be verified and from reliable sources.

                IMPORTANT: Do not use any markdown formatting, bold text, bullet points, numbered lists, or special characters. 
                Respond in plain, natural speech only. Keep responses very concise for voice conversation.
            """
        )

        await mainAgent.finish()

launcher = AgentLauncher(
    join_call = join_call,
    max_sessions_per_call = 1,
    create_agent = create_agent
)

runner = Runner(launcher)

@runner.fast_api.post("/api/stream/agent/join")
async def join(data: JoinRequestData):
    call_id = data.call_id

    agent_configs[call_id] = {
        "id": data.id,
        "name": data.name,
        "image": data.image,
        "instructions": data.instructions
    }

    await launcher.start_session(call_id = call_id, call_type = data.call_type)
    return { "message": "Agent đang được kết nối với cuộc họp . . ." }

if __name__ == "__main__":
    runner.cli()