import os
import yaml
import torch
from tqdm import tqdm
from pydub import AudioSegment
# from pyannote.audio import Pipeline
from pydub.effects import split_on_silence


def open_configs(configs:list)->list:
    opened_configs = []
    for config in configs:
        with open(f'configs/{config}.yaml', 'r') as f:
            opened_configs.append(yaml.load(f.read(), Loader=yaml.FullLoader))

    return opened_configs


def split_rttm_file(rttm_file_path, wav_filepath, out_dir, remove_overlap=True):
    os.makedirs(f'{out_dir}/speaker_1', exist_ok=True)
    os.makedirs(f'{out_dir}/speaker_2', exist_ok=True)
    with open(rttm_file_path, 'r') as f:
        file = f.readlines()
    current_speaker = ''
    total_time_chunk = 0
    initial_start_time = 0
    audio_segment = AudioSegment.from_file(wav_filepath) 
    start_idx = 0 # For removing overlaps
    for i, line in enumerate(tqdm(file)):
        line = line.split(" ")
        if i == 0:
            start_idx = 0
            initial_start_time = float(line[3])
            total_time_chunk += float(line[4])
            current_speaker = line[7]
            speaker_1 = current_speaker
        else:
            start_time = float(line[3])
            duration = float(line[4])
            speaker = line[7]

            if speaker == current_speaker:
                total_time_chunk += duration
            else:
                if total_time_chunk * 1000 > 1250:
                    # Export previous clips
                    start = int(initial_start_time*1000)
                    end = start + int(total_time_chunk*1000)
                    if remove_overlap:
                        prev_time_len = int((float(file[start_idx-1].split(" ")[3]) + float(file[start_idx-1].split(" ")[4])) * 1000)
                        next_start_time = int(float(file[i+1].split(" ")[3]) * 1000)
                        if prev_time_len > start:
                            start = prev_time_len
                        if next_start_time < end:
                            end = next_start_time
                    # if end < start:
                        # continue
                    if speaker == speaker_1:
                        chunk_idx = len(os.listdir(f'{out_dir}/speaker_1')) + 1
                        audio_segment[start:end].export(f'{out_dir}/speaker_1/{chunk_idx}.wav', format='wav')
                    else:
                        chunk_idx = len(os.listdir(f'{out_dir}/speaker_2')) + 1
                        audio_segment[start:end].export(f'{out_dir}/speaker_2/{chunk_idx}.wav', format='wav')
                # Set new values
                initial_start_time = start_time
                total_time_chunk = duration
                current_speaker = speaker
                start_idx = i

    
def split_audio_on_silence(config, file_path):
    audio = AudioSegment.from_file(file_path)
    audio_chunks = split_on_silence(audio, 
                                    min_silence_len=config['min_silence_len'],
                                    keep_silence=config['keep_silence'],
                                    silence_thresh=config['silence_thresh'])
    fname = file_path.split("/")[-1]
    os.makedirs(f'data/processed_data/{fname.split(".")[0]}', exist_ok=True)
    for i, chunk in enumerate(audio_chunks):
        chunk.export(f'data/processed_data/{fname.split(".")[0]}/{i}_{fname}', format='wav')


def diarize_speakers(audio_path, access_token):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=access_token).to(device)

    # apply the pipeline to an audio file
    diarization = pipeline(audio_path)
    # dump the diarization output to disk using RTTM format
    with open(f'{audio_path.split(".")[0]}.rttm', "w") as rttm:
        diarization.write_rttm(rttm)

