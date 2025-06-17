import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import MCPServerAdapter


@CrewBase
class McpTest:
    """McpTest crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    mcp_server = None

    def __init__(self):
        server_params = {
            "url": os.getenv("MCP_URL", ""),
            "transport": os.getenv("MCP_TRANSPORT"),
        }
        self.mcp_server = MCPServerAdapter(server_params)
        self.tools = self.mcp_server.tools

        print(
            f">>> Available tools from the MCP server: {[tool.name for tool in self.tools]}"
        )

    @agent
    def math_teacher(self) -> Agent:
        return Agent(
            config=self.agents_config["math_teacher"],  # type: ignore[index]
            verbose=True,
            tools=self.tools,
        )

    @task
    def solve_math_problem(self) -> Task:
        return Task(
            config=self.tasks_config["solve_math_problem"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the McpTest crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
