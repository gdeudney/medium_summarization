import lmstudio as lms

model = lms.llm("qwen3-32b")
chat = "Tell me about Canada"
result = model.respond(chat, config={
    "temperature": 0.6,
})


print(result)