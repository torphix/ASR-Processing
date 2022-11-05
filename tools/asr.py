import os
import torch
import shutil
import whisper
import librosa
from tqdm import tqdm
from multiprocessing import Pool

class ASR:
    def __init__(self, config):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model(config['model_type'], device=device)
        self.input_path = config['input_path']

    def transcribe(self):
        if os.listdir(self.input_path)[0].endswith('wav'):
            multispeaker = False
        else:
            multispeaker = True
        # Get files
        files = []
        if multispeaker:
            for speaker in os.listdir(self.input_path):
                for file in os.listdir(f'{self.input_path}/{speaker}'):
                    if file.endswith('wav'):
                        files.append(f'{self.input_path}/{speaker}/{file}')
        else:
            for file in os.listdir(f'{self.input_path}'):
                if file.endswith('wav'):
                    files.append(f'{self.input_path}/{file}')

        for file in tqdm(files):
            self._inference(file) 
        # # Try with multiprocessing
        # try:
        #     with Pool(2) as pool:
        #         pool.map(self._inference, files)
        # Process normally
        # except:

    def _inference(self, input_path):
        # try:
        # skipped_files = 0
        # wav, sr = librosa.load(input_path)
        # if sr*self.config['max_wav_len'] > wav.shape[0]:
        #     skipped_files += 1
        #     return
        # if os.path.exists(f'{input_path.split(".")[0]}.txt'):
        #     return
        # del wav
        
        result = self.model.transcribe(input_path)
        with open(f'{input_path.split(".")[0]}.txt', 'w') as f:
            f.write(result['text'])
        # print(f'Skipped {skipped_files} files as longer than max_wav_len')
        # except:
        #     if self.config['delete_corrupted_audio']:
        #         os.remove(input_path)