## Tools for creating high quality TTS datasets
### Features
- Scrape youtube data
- Seperate speakers
- Split on silence
- High quality transcription with OpenAI whisper model

### How to use
- Go to configs
- Scrape videos from channel:
    - Enter channel link and number of videos to scrape
    - use command python main.py scrape_data --postprocess (Optional)
- Scrape videos by individual links:
    - Provide a list like object to scraper > links
    - use command python main.py scrape --postprocess (Optional)
- Respond to various audio cleaning requests
- Ensure processing parameters are to your liking in configs/config.yaml (though defaults should suffice)

- Can use individual commands on directories / files
    - python manage.py split_audio_on_silence --path={dir_or_file}
    - python manage.py transcribe --path={dir_or_file}
    - python manage.py diarize --path={dir_or_file} --n_speakers={num_speakers}
    