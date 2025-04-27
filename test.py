from openai import OpenAI

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
)

# Hỏi người dùng nhập prompt
user_prompt = input("Nhập câu hỏi của bạn: ")

# Gửi yêu cầu tới model
completion = client.chat.completions.create(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    messages=[
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096,
    frequency_penalty=0,
    presence_penalty=0,
    stream=True,
)

# In kết quả stream ra
print("\n=== Trả lời ===")
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n")