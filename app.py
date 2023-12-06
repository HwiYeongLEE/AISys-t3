import gradio as gr
import librosa
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration, PreTrainedTokenizerFast, BartForConditionalGeneration, pipeline
from pytube import YouTube
import tempfile
from generator import Generator

transcription_checkpoint = "byoussef/whisper-large-v2-Ko"

checkpoint = "zzrng76/AISYSTEM"
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
model = BartForConditionalGeneration.from_pretrained(checkpoint)

assert torch.cuda.is_available()

title = "❓퀴즈 마스터 - 강의 내용으로 생성된 퀴즈를 풀어보세요!"
article = "Made by Seoultech AISys Team 3<br>2gnldud@gmail.com, semin2k1@gmail.com, yerang@seoultech.ac.kr"

transcriber = pipeline(
  "automatic-speech-recognition",
  model=transcription_checkpoint,
  chunk_length_s=30,
  device="cuda:0",
  )

generater = Generator(model, tokenizer, device="cuda:1") 

def process_audio(sampling_rate, waveform):
    waveform = waveform / 32678.0
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.T)
    if sampling_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)
    text = transcriber(waveform)["text"]
    return text

def process_youtube(url):
    with tempfile.TemporaryDirectory() as temp_dir:
       
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        try:
            temp_audio_path = audio_stream.download(output_path=temp_dir, filename="youtube_audio.wav")
        except Exception as e:
            print(f"Error downloading YouTube audio: {e}")
        text = transcriber(temp_audio_path)["text"]
        
    return text

def pipe(audio_or_url):
    if isinstance(audio_or_url, tuple):
        # Input is audio file
        output = process_audio(*audio_or_url)
    elif isinstance(audio_or_url, str):
        # Input is YouTube URL
        output = process_youtube(audio_or_url)
    else:
        raise ValueError("Invalid input format")
    
    kwargs= {
        'frame_size':128,
        'hop_length':64,
        'max_length':512,
        'num_beams':6
    }

    output = generater.make_question(output, **kwargs)
    output = "\n".join(output)

    return output

demo = gr.Blocks()

from_file = gr.Interface(
    title=title,
    fn=pipe,
    inputs=[
        gr.Audio(sources="upload", type="numpy", label="Audio file"),
    ],
    outputs=[
        gr.Textbox(label='Generated Question', lines=10)
    ],
    theme="huggingface",
    allow_flagging="never",
    article=article
)

from_youtube = gr.Interface(
    title=title,
    fn=pipe,
    inputs=[
        gr.Textbox(label='YouTube URL', type='text'),
    ],
    outputs=[
        gr.Textbox(label='Generated Question', lines=10)
    ],
    theme="huggingface",
    allow_flagging="never",
    article=article
)

with demo:
    gr.TabbedInterface([from_file, from_youtube], ["Audio file", "YouTube"])

demo.launch(share=True)






#Legacy Code

# def transcribe(audio):
#     waveform = process_audio(*audio)
#     input_features = transcription_processor(
#         waveform, sampling_rate=16000, return_tensors="pt"
#     ).input_features.to("cuda:0")

#     generated_ids = transcription_model.generate(inputs=input_features)
#     transcription = transcription_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(transcription)
#     return transcription

# def generate_question(text):
#     input_ids = [generation_tokenizer.bos_token_id] + generation_tokenizer.encode(text) + [generation_tokenizer.eos_token_id]
#     input_ids = torch.tensor([input_ids]).to("cuda:1")
#     res_ids = generation_model.generate(
#         input_ids,
#         max_length=512,
#         num_beams=3,
#         eos_token_id=generation_tokenizer.eos_token_id,
#         bad_words_ids=[[generation_tokenizer.unk_token_id]]
#     )
#     res_ids = res_ids.to(device="cpu")
#     a = generation_tokenizer.batch_decode(res_ids.tolist())[0]
#     out_question = a.replace('<s>', '').replace('</s>', '')
#     return out_question