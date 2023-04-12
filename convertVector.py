import music21
import numpy as np
import matplotlib.pyplot as plt
import mido
import csv
import IPython.display as ipd
import midi2audio
import glob
import convertVector as cv

class convertVector:
    TOTAL_MEASURES = 240        # 学習用MusicXMLを読み込む際の小節数の上限
    UNIT_MEASURES = 4           # 1回の生成で扱う旋律の長さ
    BEAT_RESO = 4               # 1拍を何個に分割するか（4の場合は16分音符単位）
    N_BEATS = 4                 # 1小節の拍数（今回は4/4なので常に4）
    NOTENUM_FROM = 36           # 扱う音域の下限（この値を含む）
    NOTENUM_THRU = 84           # 扱う音域の上限（この値を含まない）
    INTRO_BLANK_MEASURES = 4    # ブランクおよび伴奏の小節数の合計
    MELODY_LENGTH = 8           # 生成するメロディの長さ（小節数）
    KEY_ROOT = "C"              # 生成するメロディの調のルート（"C" or "A"）
    KEY_MODE = "major"          # 生成するメロディの調のモード（"major" or "minor"）

    def setKeyRoot(self, root):
        self.KEY_ROOT = root
    def setKeyMode(self, mode):
        self.KEY_MODE = mode
        

    # Note列をone-hot vector列（休符はすべて0）に変換
    def note_seq_to_onehot(self, note_seq):
        M = self.NOTENUM_THRU - self.NOTENUM_FROM
        N = len(note_seq)
        matrix = np.zeros((N, M))
        for i in range(N):
            if note_seq[i] != None:
                matrix[i, note_seq[i].pitch.midi - self.NOTENUM_FROM] = 1
        return matrix


    # 音符列を表すone-hot vector列に休符要素を追加
    def add_rest_nodes(self, onehot_seq):
        rest = 1 - np.sum(onehot_seq, axis=1)
        rest = np.expand_dims(rest, 1)
        return np.concatenate([onehot_seq, rest], axis=1)


    # 指定された仕様のcsvファイルを読み込んで
    # ChordSymbol列を返す
    def read_chord_file(self, file):
        chord_seq = [None] * (self.MELODY_LENGTH * self.N_BEATS)
        with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                m = int(row[0]) # 小節番号（0始まり）
                if m < self.MELODY_LENGTH:
                    b = int(row[1]) # 拍番号（0始まり、今回は0または2）
                    chord_seq[m*4+b] = music21.harmony.ChordSymbol(root=row[2], 
                                                                    kind=row[3], 
                                                                    bass=row[4])
            for i in range(len(chord_seq)):
                if chord_seq[i] != None:
                    chord = chord_seq[i]
                else:
                    chord_seq[i] = chord
        return chord_seq


    # コード進行からChordSymbol列を生成
    # divisionは1小節に何個コードを入れるか
    def make_chord_seq(self, chord_prog, division):
        T = int(self.N_BEATS * self.BEAT_RESO / division)
        seq = [None] * (T * len(chord_prog))
        for i in range(len(chord_prog)):
            for t in range(T):
                if isinstance(chord_prog[i], music21.harmony.ChordSymbol):
                    seq[i * T + t] = chord_prog[i]
                else:
                    seq[i * T + t] = music21.harmony.ChordSymbol(chord_prog[i])
        return seq


    # ChordSymbol列をmany-hot (chroma) vector列に変換
    def chord_seq_to_chroma(self, chord_seq):
        N = len(chord_seq)
        matrix = np.zeros((N, 12))
        for i in range(N):
            if chord_seq[i] != None:
                for note in chord_seq[i]._notes:
                    matrix[i, note.pitch.midi % 12] = 1
        return matrix

        
    # メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から
    # モデルの入力、出力用のデータを作成して、配列に逐次格納する
    def divide_seq(self, onehot_seq, chroma_seq, x_all, y_all):
        for i in range(0, self.TOTAL_MEASURES, self.UNIT_MEASURES):
            o, c, = self.extract_seq(i, onehot_seq, chroma_seq)
            if np.any(o[:, 0:-1] != 0):
                x, y = self.calc_xy(o, c)
                x_all.append(x)
                y_all.append(y)
    def calc_xy(self, o, c):
        #x = np.concatenate([o, c], axis=1)
        x = o
        y = o
        return x, y
    # メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列に対して、
# UNIT_MEASURES小節分だけ切り出したものを返す
    def extract_seq(self, i, onehot_seq, chroma_seq):
        o = onehot_seq[i*self.N_BEATS* self.BEAT_RESO : (i+ self.UNIT_MEASURES)* self.N_BEATS* self.BEAT_RESO, :]
        c = chroma_seq[i* self.N_BEATS* self.BEAT_RESO : (i+ self.UNIT_MEASURES)* self.N_BEATS* self.BEAT_RESO, :]
        return o, c
