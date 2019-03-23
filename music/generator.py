from __future__ import unicode_literals, print_function, division
import mido
from mido import MidiFile
from mido import Message
from mido import Message, MidiFile, MidiTrack
from operator import itemgetter
import random
import timeit
import time
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from midi2audio import FluidSynth
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 100000000
max_dur_note = 4
amount_dur = 10
each_dur = 0.2
hidden_size = 256
list_indexes = []
duration = 60
count = 16
SOS_token = 128 * amount_dur + 1
EOS_token = 128 * amount_dur + 2


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def midi_from_indexes(indexes, max_dur_note, amount_dur, each_dur):
    time_now = 0
    events = []
    for i in range(len(indexes)):
        if (indexes[i] == '<EOS>'):
            note = -1
        elif (indexes[i] == 128 * amount_dur):
            note = -1
        else:
            note = indexes[i] % 128

            type_dur = (indexes[i] - note) / 128
            dur = max_dur_note / amount_dur * (type_dur + 1)
        if (note != -1):
            events.append([time_now, 'note_on', note])
            events.append([time_now + dur, 'note_off', note])
        time_now += each_dur
    events = sorted(events, key=itemgetter(0))
    for i in range(len(events) - 1, 0, -1):
        events[i][0] -= events[i - 1][0]
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for ev in events:
        track.append(Message(ev[1], note=ev[2], velocity=64, time=int(ev[0] / 0.0013354687499999999)))
    return mid


def evaluate(input_rand, decoder, max_length=MAX_LENGTH, most_prob=5):
    with torch.no_grad():

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        input_tensor = torch.FloatTensor(input_rand)
        decoder_hidden = input_tensor

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(most_prob)
            i = random.randint(0, most_prob - 1)

            topv = topv[0][i]
            topi = topi[0][i]

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())
            decoder_input = topi.squeeze().detach()

        return decoded_words


def get_sample():
    print("Enter seed in the range 1-25")
    seed = int(input())
    decoder = torch.load(os.getcwd() + '/models_Sonya/decoder_1').to(device)
    input_rand = []
    for i in range(0, hidden_size):
        input_rand.append(seed)
    input_rand = np.reshape(input_rand, (1, 1, hidden_size))
    new_indexes = evaluate(input_rand, decoder, max_length=300, most_prob=15)
    mid_new = midi_from_indexes(new_indexes[20:], max_dur_note, amount_dur, each_dur)
    mid_new.save('new_song.mid')
    print('OK')
    # using the default sound font in 44100 Hz sample rate
    fs = FluidSynth(os.getcwd() + '/TimGM6mb.sf2')
    fs.midi_to_audio('new_song.mid', 'new_song.wav')



get_sample()