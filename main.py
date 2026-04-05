from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from vision_agents.plugins import getstream, gemini
from vision_agents.core import Agent, AgentLauncher, Runner, User

load_dotenv()

agent_configs = {}

class JoinRequest(BaseModel):
    id: str
    name: str
    image: str
    call_id: str
    call_type: str
    instructions: str

async def create_agent(**kwargs) -> Agent:
    return Agent(
        agent_user = User(
            id = "id",
            name = "Agent"
        ),
        llm = gemini.Realtime(),
        edge = getstream.Edge()
    )

async def join_call(agent, call_type, call_id, **kwargs):
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
        await mainAgent.simple_response("Chào các thành viên trong cuộc họp, tóm tắt nhiệm vụ của cuộc họp và trách nhiệm của bạn trong cuộc họp. Cuối cùng yêu cầu người dùng đặt câu hỏi để bắt đầu cuộc trò chuyện.")
        await mainAgent.finish()

runner = Runner(AgentLauncher(create_agent = create_agent, join_call = join_call))
runner.fast_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@runner.fast_api.post("/custom/join")
async def custom_join(body: JoinRequest):
    call_id = body.call_id

    agent_configs[call_id] = {
        "image": body.image,
        "name": body.name,
        "id": body.id,
        "instructions": body.instructions
    }

    await runner._launcher.start_session(call_id = call_id, call_type = body.call_type)
    return { "message": "Agent đang được kết nối với cuộc họp . . ." }

if __name__ == "__main__":
    runner.cli()