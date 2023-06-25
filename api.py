import asyncio
import traceback
from datetime import datetime
import numpy as np
from flask import Flask, request
from flask_cloudflared import run_with_cloudflared
import os
import glob
import json
import traceback
import logging
import gradio as gr
import numpy as np
import librosa
import torch
import asyncio
import edge_tts
import yt_dlp
import ffmpeg
import subprocess
import sys
import io
import wave
from datetime import datetime
from fairseq import checkpoint_utils
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from vc_infer_pipeline import VC
from config import Config
from scipy.io import wavfile
import base64
import json

config = Config()
logging.getLogger("numba").setLevel(logging.WARNING)

def create_vc_fn(tgt_sr, net_g, vc, if_f0, file_index):
    def vc_fn(
        input_audio,
        upload_audio,
        upload_mode,
        f0_up_key,
        f0_method,
        index_rate,
        tts_mode,
        tts_text,
        tts_voice
    ):
        try:
            if tts_mode:
                if tts_text is None or tts_voice is None:
                    return "You need to enter text and select a voice", None
                asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save("tts.mp3"))
                audio, sr = librosa.load("tts.mp3", sr=16000, mono=True)
            else:
                if upload_mode:
                    if input_audio is None:
                        return "You need to upload an audio", None
                    sampling_rate, audio = upload_audio
                    duration = audio.shape[0] / sampling_rate
                    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio.transpose(1, 0))
                    if sampling_rate != 16000:
                        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio, sr = librosa.load(input_audio, sr=16000, mono=True)
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                if_f0,
                f0_file=None,
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            )
            return (tgt_sr, audio_opt)
        except:
            info = traceback.format_exc()
            print(info)
            return info, (None, None)
    return vc_fn

def cut_vocal_and_inst(url, audio_provider, split_model):
    if url != "":
        if not os.path.exists("dl_audio"):
            os.mkdir("dl_audio")
        if audio_provider == "Youtube":
            ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            "outtmpl": 'dl_audio/youtube_audio',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            audio_path = "dl_audio/youtube_audio.wav"
        else:
            # Spotify doesnt work.
            # Need to find other solution soon.
            ''' 
            command = f"spotdl download {url} --output dl_audio/.wav"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE)
            print(result.stdout.decode())
            audio_path = "dl_audio/spotify_audio.wav"
            '''
        if split_model == "htdemucs":
            command = f"demucs --two-stems=vocals {audio_path} -o output"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE)
            print(result.stdout.decode())
            return "output/htdemucs/youtube_audio/vocals.wav", "output/htdemucs/youtube_audio/no_vocals.wav", audio_path, "output/htdemucs/youtube_audio/vocals.wav"
        else:
            command = f"demucs --two-stems=vocals -n mdx_extra_q {audio_path} -o output"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE)
            print(result.stdout.decode())
            return "output/mdx_extra_q/youtube_audio/vocals.wav", "output/mdx_extra_q/youtube_audio/no_vocals.wav", audio_path, "output/mdx_extra_q/youtube_audio/vocals.wav"
    else:
        raise gr.Error("URL Required!")
        return None, None, None, None

def combine_vocal_and_inst(audio_data, audio_volume, split_model):
    if not os.path.exists("output/result"):
        os.mkdir("output/result")
    vocal_path = "output/result/output.wav"
    output_path = "output/result/combine.mp3"
    if split_model == "htdemucs":
        inst_path = "output/htdemucs/youtube_audio/no_vocals.wav"
    else:
        inst_path = "output/mdx_extra_q/youtube_audio/no_vocals.wav"
    with wave.open(vocal_path, "w") as wave_file:
        wave_file.setnchannels(1) 
        wave_file.setsampwidth(2)
        wave_file.setframerate(audio_data[0])
        wave_file.writeframes(audio_data[1].tobytes())
    command =  f'ffmpeg -y -i {inst_path} -i {vocal_path} -filter_complex [1:a]volume={audio_volume}dB[v];[0:a][v]amix=inputs=2:duration=longest -b:a 320k -c:a libmp3lame {output_path}'
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode())
    return output_path

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def change_to_tts_mode(tts_mode, upload_mode):
    if tts_mode:
        return gr.Textbox.update(visible=False), gr.Audio.update(visible=False), gr.Checkbox.update(visible=False), gr.Textbox.update(visible=True), gr.Dropdown.update(visible=True)
    else:
        if upload_mode:
            return gr.Textbox.update(visible=False), gr.Audio.update(visible=True), gr.Checkbox.update(visible=True), gr.Textbox.update(visible=False), gr.Dropdown.update(visible=False)
        else:
            return gr.Textbox.update(visible=True), gr.Audio.update(visible=False), gr.Checkbox.update(visible=True), gr.Textbox.update(visible=False), gr.Dropdown.update(visible=False)

def change_to_upload_mode(upload_mode):
    if upload_mode:
        return gr.Textbox().update(visible=False), gr.Audio().update(visible=True)
    else:
        return gr.Textbox().update(visible=True), gr.Audio().update(visible=False)


load_hubert()
categories = []
tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
with open("weights/folder_info.json", "r", encoding="utf-8") as f:
    folder_info = json.load(f)
for category_name, category_info in folder_info.items():
    if not category_info['enable']:
        continue
    category_title = category_info['title']
    category_folder = category_info['folder_path']
    description = category_info['description']
    models = []
    with open(f"weights/{category_folder}/model_info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)
    for model_name, info in models_info.items():
        if not info['enable']:
            continue
        model_title = info['title']
        model_author = info.get("author", None)
        model_cover = f"weights/{category_folder}/{model_name}/{info['cover']}"
        model_index = f"weights/{category_folder}/{model_name}/{info['feature_retrieval_library']}"
        cpt = torch.load(f"weights/{category_folder}/{model_name}/{model_name}.pth", map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(config.device)
        if config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, config)
        print(f"Model loaded: {model_name}")
        models.append((model_name, model_title, model_author, model_cover, create_vc_fn(tgt_sr, net_g, vc, if_f0, model_index)))
    categories.append([category_title, category_folder, description, models])
    for (folder_title, folder, description, models) in categories:
        for (name, title, author, cover, vc_fn) in models:
            app = Flask(__name__)
            run_with_cloudflared(app)  # Open a Cloudflare Tunnel when app is run
            @app.route('/')
            def index():
                audio_file = '/content/output.wav'  # Path to your audio file
                return render_template('index.html', audio_file=audio_file)
                
            @app.route('/api/vc', methods=['POST'])
            def vc_api():
                # Retrieve the data from the POST request
                data = request.get_json()
            
                # Call the vc_fn function with the provided data
                result = vc_fn(
                    data['input_audio'],
                    data['upload_audio'],
                    data['upload_mode'],
                    data['f0_up_key'],
                    data['f0_method'],
                    data['index_rate'],
                    data['tts_mode'],
                    data['tts_text'],
                    data['tts_voice']
                )
                output_file = '/content/output.wav'

                # Write the audio data to a WAV file
                wavfile.write(output_file, result[0], result[1])

                # Read the audio file
                with open(output_file, 'rb') as file:
                    audio_data = file.read()
                
                # Encode the audio data as base64
                encoded_audio = base64.b64encode(audio_data).decode('utf-8')

                # Return the result as a JSON response
                # return {
                #         'message': result[0],
                #         'audio': result[1].tolist()  # Convert the ndarray to a nested Python list
                #     }

                response_data = {
                        'audio': encoded_audio
                    }
                # Convert the response to JSON
                json_response = json.dumps(response_data)
                
                # Return the JSON response
                return json_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
