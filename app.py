import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
# model = model.to_bettertransformer()
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe(task, file):
        args = {'task': task}

        if file is not None:
            audio = file
        else:
            return "You must provide a mic recording or a file"
        result = pipe(audio, **args)
        return result["text"]



demo = gr.Interface(transcribe, 
    inputs=[
        gr.Radio(['transcribe', 'translate'], label= 'Task'),
        gr.Audio(sources=["upload"], type="filepath", optional=True, label='Audio File')
        ], 
    outputs="text")
demo.launch(share=True) 