from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003")

txt = open("data/keynote.txt").read()
task = "download this `article` to a local file?"
print(f"""==The task==
{task}

""")

res = agent.run(
    task,
    article="https://www.fastcompany.com/90890638/hallucinations-data-leaks-toxic-language-arthur-ai-chatgpt-generative-ai",
    return_code=False,
    remote=True,
    lables=['article']
)

print(res)
