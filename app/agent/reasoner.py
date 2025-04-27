from openai import OpenAI
import os

class Reasoner:
    def __init__(self):
        # Khởi tạo client với API key và base_url
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("api_key")
        )

    def plan(self, prompt):
        # Gửi yêu cầu đến API để lên kế hoạch trả lời
        completion = self.client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[{"role": "system", "content": f"Plan how to answer this prompt: {prompt}"}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        # Nhận và trả về kết quả từ API
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response

    def refine(self, plan, file_context, web_context):
        # Gửi yêu cầu đến API để làm rõ kế hoạch
        completion = self.client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[{
                "role": "system",
                "content": f"Refine this plan: {plan} using data: {file_context} and {web_context}"
            }],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        # Nhận và trả về kết quả từ API
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response

    def reflect(self, report):
        # Gửi yêu cầu đến API để cải thiện báo cáo
        completion = self.client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[{
                "role": "system",
                "content": f"Review and improve this report: {report}"
            }],
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )
        # Nhận và trả về kết quả từ API
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response
