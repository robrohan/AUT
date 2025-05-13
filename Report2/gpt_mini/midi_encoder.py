import pretty_midi
import numpy as np
from typing import Any, Tuple

# 24, 48, 96, 120, 240, 384, 480, and 960
COMMON_RESOLUTION=240

def encode_header(key: int, bpm: int, nominator: int, denominator:int ) -> int:
    """
    |           | KEY       |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |

    | BPM       | TIME/SIG  |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |
    """
    encoded_header = 0
    encoded_header += (key&255) << 16
    encoded_header += (bpm&255) << 8
    encoded_header += (nominator&15) << 4
    encoded_header += (denominator&15)
    # print(f"{encoded_header:32b}")
    return encoded_header


def decode_header(header: int) -> Tuple[int,int,int,int]:
    """
    |           | KEY       |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |

    | BPM       | TIME/SIG  |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |
    """
    key = (header >> 16) & 255
    bpm = (header >> 8) & 255
    nominator = (header >> 4) & 15
    denominator = header & 15
    return (int(key), int(bpm), int(nominator), int(denominator))


def encode_note(note: Any, ticks_per_beat: int) -> int:
    if note is None:
        0
    # convert seconds into ticks so we can use a common tick
    start = int(note.start * ticks_per_beat)
    end = int(note.end * ticks_per_beat)
    velocity = note.velocity
    pitch = note.pitch

    int_velocity = int( ((velocity / 127) * 15) ) & 15
    int_step = start & 65535
    int_duration = (end - start) & 31

    encoded_note = 0
    encoded_note += int_step << 16
    encoded_note += int_duration << 11
    encoded_note += int_velocity << 7
    encoded_note += pitch

    # print(f"{encoded_note:32b}")
    return encoded_note


def decode_note(encoded_note: int, ticks_per_beat: int) -> Tuple:
    pitch = (encoded_note) & 127
    velocity = (encoded_note >> 7) & 15
    int_step = (encoded_note >> 16) & 65535
    int_duration = (encoded_note >> 11) &31

    # print(f"{encoded_note:32b}")
    f_step = int_step / ticks_per_beat
    f_duration = int_duration / ticks_per_beat
    f_duration = f_step+f_duration
    velocity = int((velocity/15)*127)

    return (pitch&127, f_step, f_duration, velocity&127)


def extract_midi_metadata(pm) -> Tuple[int,int,int,int]:
    bpm = 120
    key_signature = 0
    nominator = 4
    denominator = 4

    if pm.get_tempo_changes():
        bpm = pm.get_tempo_changes()[1][0]

    if pm.key_signature_changes:
        key_signature = pm.key_signature_changes[0].key_number

    if pm.time_signature_changes:
        time_signature = pm.time_signature_changes[0]
        nominator = time_signature.numerator
        denominator = time_signature.denominator

    return (int(key_signature), int(bpm), int(nominator), int(denominator))


def encode_midi(midi_file: str,
                window_size=64,
                instrument_name: str = "Standard Kit") -> np.array:
    """
    Encode a midi file into a numpy array of integers using the
    instrument index with a max window size of window_size
    (window size is basically the number of note on events)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        ticks_per_beat = pm.resolution
        instrument = None

        data = extract_midi_metadata(pm)
        header = encode_header(data[0], data[1], data[2], data[3])
        # print(data)

        instrument_index = None
        if instrument_name != "Standard Kit":
            instrument_index = pretty_midi.instrument_name_to_program(instrument_name)

        if instrument_index is not None:
            instrument = pm.instruments[instrument_index]
        else:
            # Collect all the drum tracks (sometimes they are on
            # different track, kick and snare vs cymbals for example)
            # drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
            for inst in pm.instruments:
                if inst.is_drum:
                    instrument = inst
            #         for note in inst.notes:
            #             drum_instrument.notes.append(note)
            # instrument = drum_instrument

        notes = np.zeros(window_size, dtype=np.uint32, order='C')
        notes[0] = header
        # sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        sorted_notes = instrument.notes

        for i, note in enumerate(sorted_notes):
            if i >= (window_size-1):
                break
            encoded_note = encode_note(note, COMMON_RESOLUTION)
            notes[i+1] = encoded_note

    except Exception as e:
        print(f"could not load {midi_file} because {e}")
        return None

    return np.array(notes)


def decode_midi(
    notes: np.array,
    out_file: str,
    instrument_name: str = "Standard Kit") -> pretty_midi.PrettyMIDI:
    """
    Given an array of encoded midi integers write a new midi file using
    the general midi instrument name as the instrument to use.

    Key number according to [0, 11] Major, [12, 23] minor. For example, 0
    is C Major, 12 is C minor
    """

    header = notes[0]
    data = decode_header(header)
    key = data[0]
    bpm = data[1]
    numerator = data[2]
    denominator = data[3]

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name) if instrument_name != "Standard Kit" else 0,
        is_drum=True if instrument_name == "Standard Kit" else False,
    )

    time_signature = pretty_midi.TimeSignature(numerator=numerator, denominator=denominator, time=0)
    pm.time_signature_changes.append(time_signature)
    key_signature = pretty_midi.KeySignature(key_number=key, time=0)
    pm.key_signature_changes.append(key_signature)

    # print(bpm, pm.resolution)
    # 24, 48, 96, 120, 240, 384, 480, and 960 ticks per beat
    # tick = 60 / (bpm * pm.resolution)
    tick = 60 / (bpm * COMMON_RESOLUTION)
    pm._tick_scales.append((0, tick))
    pm._update_tick_to_time(0)
    # ticks_per_beat = pm.resolution

    pm.instruments.append(instrument)

    for i, encoded_note in enumerate(notes):
        if i == 0:
            # Skip header
            continue
        pitch, start, end, velocity = decode_note(encoded_note, COMMON_RESOLUTION)
        note = pretty_midi.Note(
            pitch=pitch,
            start=start,   # should be in seconds not ticks
            end=end,       # should be in seconds not ticks
            velocity=velocity,
        )
        instrument.notes.append(note)

    pm.write(out_file)
    return pm


def serialize_notes(raw_notes: np.array, filename: str):
    raw_notes.astype('int32').tofile(filename)


def deserialize_notes(filename: str) -> np.array:
    return np.fromfile(filename, dtype=np.int32)
