import gradio as gr
from chat import response

# Replace with your API endpoint

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "

def chatgpt_clone(input, history):
    history = history or []
    # print(history)
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = response(inp)
    history.append((input, output))
    return history, history

block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>Research on Developing Application for Legal Question Answering Using LLM and RAG</center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug=True)
