import re
import os
import pytube
from tqdm import tqdm
from pytube import Channel
from pydub import AudioSegment
from multiprocessing import Pool
from pydub.silence import split_on_silence


class Scraper:
    '''
    - Scrapes youtube vidoes
    - Splits on silence
    - Transcribe audio clips
    '''
    def __init__(self, config, n_workers):
        self.config = config
        self.n_workers = n_workers

    def scrape_channel(self):
        print('Scraping Channel...')
        channel = Channel(self.config['channel_link'])
        if self.config['scrape_n_videos_from_channel'] == -1:
            channel_videos = list(channel.video_urls)
        else:
            channel_videos = list(channel.video_urls)[:self.config['scrape_n_videos_from_channel']]

        if len(channel_videos) == 0:
            raise Exception('''
                Channel has no videos OR bug related to https://github.com/pytube/pytube/issues/1408
                Input individual links for now or try updating pytube
                ''')

        self.channel_name = re.split('\W+', channel.channel_name)[0]
        link_zip = zip([(vid,f'{self.channel_name.lower()}_{i}') 
                        for i,vid in enumerate(channel_videos)])
        with Pool(processes=self.n_workers) as pool:
            output_file_paths = list(tqdm(pool.starmap(self._scrape_link, link_zip), 
                                          total=len(channel_videos)))
        return output_file_paths

    def scrape_links(self):
        print('Scraping Links...')
        link_zip = zip(self.config['links'], [f'link_{i}' for i in range(len(self.config['links']))])
        with Pool(processes=self.n_workers) as pool:
            output_file_paths = list(tqdm(pool.starmap(self._scrape_link, link_zip), 
                                          total=len(self.config['links'])))
        return output_file_paths

    def _scrape_link(self, link, fname):
        yt = pytube.YouTube(link)
        audio_stream = yt.streams.filter(only_audio=True).first()
        output_path = f'{self.config["raw_data_path"]}/{fname}'
        os.makedirs(output_path, exist_ok=True)
        audio_stream.download(output_path, f'{fname}.wav')
        return f'{output_path}/{fname}.wav'

    def process_audio(self, out_path, fname):
        # split_cmd_out = os.system(" ".join(['spleeter', 'separate', '-p', 'spleeter:2stems', '-o', 'output', f'{out_path}/{fname}']))
        # shutil.move(f'output/{fname.split(".")[0]}/vocals.wav', f'{out_path}/{fname.split(".")[0]}.wav')
        # os.remove(f'{out_path}/{fname}')
        # shutil.rmtree(f'output')
        fname = fname.split(".")[0] + '.wav'
        audio = AudioSegment.from_file(f'{out_path}/{fname}')
        audio_chunks = split_on_silence(audio, 
                                        min_silence_len=150,
                                        keep_silence=500,
                                        silence_thresh=-40)
        for i, chunk in enumerate(audio_chunks):
            chunk.export(f'{out_path}/{i}_{fname}', format='wav')

     