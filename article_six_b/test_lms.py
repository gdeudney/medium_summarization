import lmstudio as lms

model = lms.llm("gpt-oss-20b")
chat = "What is the meaning of life? format the response in valid JSON"
result = model.respond(chat, config={
    "temperature": 0.6,
})


print(result)