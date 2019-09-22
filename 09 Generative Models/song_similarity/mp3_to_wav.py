'''
The MP3 files in a specified directory are first converted to wav files, using
the FFMPEG converter, which must be downloaded and installed from here:
https://ffmpeg.org/
'''

from pathlib import Path
from mutagen.id3 import ID3
from pydub import AudioSegment
import csv

class MP3Processor():
    
    def __init__(self, mp3_file_dir, wav_file_dir, metadata_dir, ffmpeg_dir):
        self.mp3_file_dir = mp3_file_dir
        self.wav_file_dir = wav_file_dir
        self.metadata_dir = metadata_dir
        self.file_extensions = ("*.mp3")
        self.audiosegment = AudioSegment
        self.audiosegment.converter = ffmpeg_dir


    def getTags(self, path):
        audio = ID3(path)
        audiotags = {}
        try:
            audiotags['artist'] = audio['TPE1'].text[0]
        except:
            audiotags['artist'] = None
        try:
            audiotags['song'] = audio['TIT2'].text[0]
        except:
            audiotags['song'] = None
        try:
            audiotags['album'] = audio['TALB'].text[0]
        except:
            audiotags['album'] = None
        try:
            audiotags['year'] = audio['TDRC'].text[0]
        except:
            audiotags['year'] = None
        return audiotags

    
    def convertFiles(self):
        
        # Iterate through the mp3's in the directory, compile a list of tags and convert the files to wav
        tagslist = []
        id_tracker = 1
        file_paths = Path(self.mp3_file_dir).glob("*.mp3")
        for path in file_paths:
            file = str(path)
            mp3tags = self.getTags(file)
            mp3tags['audio_file_id'] = id_tracker
            tagslist.append(mp3tags)
            id_tracker += 1
            audio = self.audiosegment.from_mp3(file)
            audio = audio[30*1000:50*1000]  # Slice song from 30 to 50 seconds
            audio.export(self.wav_file_dir+path.stem+".wav", format="wav")

        # Write the list of dictionaries with song metadata to a csv file
        with open(self.metadata_dir+'Song_Metadata.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, tagslist[0].keys())
            w.writeheader()
            w.writerows(tagslist)

        return tagslist
