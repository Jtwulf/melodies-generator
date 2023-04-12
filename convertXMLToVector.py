import music21
import numpy as np
import matplotlib.pyplot as plt
import mido
import csv
import IPython.display as ipd
import midi2audio
import glob
from convertVector import convertVector

class xmlLoader(convertVector):
    DIR = ""
    def __init__(self, dir) :
        print("Welcome to XMLoader")
        self.DIR = dir
                

    def convert(self, root, mode):
        self.setKeyMode(mode)
        self.setKeyRoot(root)
        x_all = []
        y_all = []
        print("-----------XML LOADING -----------------")
        for f in glob.glob(self.DIR + "/*.xml"):
            print(f)
            score = music21.converter.parse(f)
            key = score.analyze("key")
            if key.mode == self.KEY_MODE:
                inter = music21.interval.Interval(key.tonic, music21.pitch.Pitch(self.KEY_ROOT))
                score = score.transpose(inter)
                note_seq, chord_seq = self.make_note_and_chord_seq_from_musicxml(score)
                onehot_seq = self.add_rest_nodes(self.note_seq_to_onehot(note_seq))
                chroma_seq = self.chord_seq_to_chroma(chord_seq)
                self.divide_seq(onehot_seq, chroma_seq, x_all, y_all)

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        print("----------------------------------------")
        return x_all, y_all
                

# MusicXMLデータからNote列とChordSymbol列を生成
# 時間分解能は BEAT_RESO にて指定
    def make_note_and_chord_seq_from_musicxml(self, score):
        note_seq = [None] * (self.TOTAL_MEASURES * self.N_BEATS * self.BEAT_RESO)
        chord_seq = [None] * (self.TOTAL_MEASURES * self.N_BEATS * self.BEAT_RESO)

        

        for element in score.parts[0].elements:
            if isinstance(element, music21.stream.Measure):
                measure_offset = element.offset
                for note in element.notes:
                    if isinstance(note, music21.note.Note):
                        onset = measure_offset + note._activeSiteStoredOffset
                        offset = onset + note._duration.quarterLength
                        for i in range(int(onset * self.BEAT_RESO), int(offset * self.BEAT_RESO + 1)):
                            note_seq[i] = note
                    if isinstance(note, music21.harmony.ChordSymbol):
                        chord_offset = measure_offset + note.offset
                        for i in range(int(chord_offset * self.BEAT_RESO), 
                            int((measure_offset + self.N_BEATS) * self.BEAT_RESO + 1)):
                            chord_seq[i] = note
        return note_seq, chord_seq
    

        
