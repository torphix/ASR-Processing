import os
import sys
import shutil
import argparse
from tqdm import tqdm
from tools.asr import ASR
from functools import partial
from multiprocessing import Pool
from tools.scraper import Scraper
from tools.utils import open_configs, diarize_speakers, split_audio_on_silence

if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'scrape':
        parser.add_argument('--workers', default=4)
        parser.add_argument("-t", "--transcribe", action='store_true')
        parser.add_argument("-s", "--split_on_silence", action='store_true')
        parser.add_argument("-m", "--merge_folders", action='store_true')
        args, lf_args = parser.parse_known_args()

        # post_process_seperate_backing_track = True if input('seperate_backing_track? (y/n)') == 'y' else False
        # post_process_diarize_speakers = True if input('diarize_speakers? (y/n)') == 'y' else False
        # if post_process_diarize_speakers:
            # diarize_access_token = input('''
                        # For speaker diarization an access token is required to use the model (it's free)
                        # Head here https://huggingface.co/pyannote/speaker-diarization fill in details
                        # Then get access token from https://huggingface.co/settings/tokens
                        # Finally paste it in here: 
                        # ''')

        config = open_configs(['config'])[0]
        scraper  = Scraper(config['scraper'], args.workers)

        if config['scraper']['links'] is not None and len(config['scraper']['links']) > 0:
            output_filepaths = scraper.scrape_links()
        if config['scraper']['channel_link'] != '' and config['scraper']['channel_link'] is not None:
            output_filepaths = scraper.scrape_channel()

        # if post_process_seperate_backing_track:
        if args.split_on_silence:
            print('Splitting audio on silence, Check config for tweakable paramters...')
            split_func = partial(split_audio_on_silence, config['post_processing']['split_on_silence'])
            with Pool(args.workers) as pool:
                list(tqdm(pool.imap(split_func, output_filepaths), total=len(output_filepaths)))
        else:
            for out in output_filepaths:
                fname = out.split("/")[-1]
                os.makedirs(f"{config['post_processing']['output_path']}/{fname.split('.')[0]}/", exist_ok=True)
                shutil.move(out, f"{config['post_processing']['output_path']}/{fname.split('.')[0]}/{fname}")

        if args.transcribe:
            asr = ASR(config['asr'])
            asr.transcribe()
            # if post_process_diarize_speakers:
            #     print('Diarizing Speaker files, Note long files take a while...')
            #     # try:
            #     #     print('Attempting with multiprocessing')
            #     #     with Pool(args.workers) as pool:
            #     #         tqdm(pool.imap(diarize_speakers, output_filepaths), total=len(output_filepaths))
            #     # except:
            #     print('Multiprocessing failed retrying linearly')
            #     for filepath in tqdm(output_filepaths):
            #         diarize_speakers(filepath, diarize_access_token)
        if args.merge_folders:
            for folder in tqdm(os.listdir(f"{config['post_processing']['output_path']}")):
                for file in os.listdir(f"{config['post_processing']['output_path']}/{folder}"):
                    shutil.move(f"{config['post_processing']['output_path']}/{folder}/{file}", f'{args.output_dir}/{file}')
                shutil.rmtree(f"{config['post_processing']['output_path']}/{folder}")
                    
    elif command == 'asr':
        args, lf_args = parser.parse_known_args()
        asr_config = open_configs(['config'])[0]['asr']
        asr = ASR(asr_config)
        asr.transcribe()

    elif command == 'clean':
        parser.add_argument('--multispeaker', action='store_true')
        parser.add_argument('--split_audio_on_silence', action='store_true')
        parser.add_argument('--seperate_backing_track', action='store_true')
        parser.add_argument('--diarize_speakers', action='store_true')
        args, lf_args = parser.parse_known_args()
        config = open_configs(['config'])        

    elif command == 'merge_folders':
        parser.add_argument('--input_dir', required=True)
        parser.add_argument('--output_dir', required=True)
        args, lf_args = parser.parse_known_args()

        for folder in tqdm(os.listdir(f'{args.input_dir}')):
            for file in os.listdir(f'{args.input_dir}/{folder}'):
                shutil.move(f'{args.input_dir}/{folder}/{file}', f'{args.output_dir}/{file}')