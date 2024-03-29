import sys
import gradio as gr

import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.deploy import build_model, preprocess_stream

model = build_model()
model.eval()


def transcribe(new_chunk_path):
    chunk_inputs, _ = preprocess_stream(new_chunk_path)
    hyps = model.model.greedy_search_streaming_app(
                chunk_inputs)
    content = []
    for w in hyps:
        if w == model.model.eos:
            break
        content.append(model.char_dict[w])
    text = model.sp.decode(content)
    text = ' '.join(text)

    return text


demo = gr.Interface(
        transcribe,
        [gr.Audio(sources=["microphone"], streaming=True, type='filepath', min_length=1)],
        ["text"],
        live=True,
    )

with demo:
    clear_btn = gr.Button(value='Reset Model')
    clear_btn.click(model.model.init_state, [], [])


demo.launch(server_port=28083, share=True)


