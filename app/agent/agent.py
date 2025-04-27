from app.agent.reasoner import Reasoner
from app.agent.retriever import Retriever
from app.agent.web_search import WebSearch
from app.agent.report_generator import ReportGenerator
from app.schemas.agent_schema import AgentRequest, AgentResponse

class Agent:
    def __init__(self):
        self.reasoner = Reasoner()
        self.retriever = Retriever()
        self.web_search = WebSearch()
        self.report_generator = ReportGenerator()

    def handle_request(self, request: AgentRequest) -> AgentResponse:
        plan = self.reasoner.plan(request.prompt)

        file_context = self.retriever.retrieve_context(request.prompt)
        web_context = self.web_search.search_web(request.prompt)

        refined_plan = self.reasoner.refine(plan, file_context, web_context)

        raw_report = self.report_generator.generate(refined_plan)

        final_report = self.reasoner.reflect(raw_report)

        return AgentResponse(
            report=final_report,
            plan=plan,
            refined_plan=refined_plan,
            file_context=file_context,
            web_context=web_context,
        )
