import pretty_midi
import numpy as np
from typing import Any, Tuple


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


def encode_note(note: Any) -> int:
    """
    | VELOCITY  | STEP      |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |

    | DURATION  | PITCH     |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |
    """
    if note is None:
        0
    start = note.start
    end = note.end
    step = start
    velocity = note.velocity
    pitch = note.pitch
    duration = end

    int_velocity = int(velocity)
    int_step = int(255 * (step/100))
    int_duration = int(255 * (duration/100))

    encoded_note = 0
    encoded_note += int_velocity << 24
    encoded_note += int_step << 16
    encoded_note += int_duration << 8
    encoded_note += pitch

    return encoded_note


def decode_note(encoded_note: int) -> Tuple:
    """
    | VELOCITY  | STEP      |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |

    | DURATION  | PITCH     |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |
    """
    pitch = (encoded_note) & 255
    velocity = (encoded_note >> 24) & 255
    int_step = (encoded_note >> 16) & 255
    int_duration = (encoded_note >> 8) & 255
    # print(f"{encoded_note:32b}")
    f_step = 100*(int_step/255)
    f_duration = 100*(int_duration/255)
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
            drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
            for inst in pm.instruments:
                if inst.is_drum:
                    #instrument = inst
                    for note in inst.notes:
                        drum_instrument.notes.append(note)
            instrument = drum_instrument

        notes = np.zeros(window_size, dtype=np.uint32, order='C')
        notes[0] = header
        # sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        sorted_notes = instrument.notes

        for i, note in enumerate(sorted_notes):
            if i >= (window_size-1):
                break
            encoded_note = encode_note(note)
            notes[i+1] = encoded_note

    except Exception as e:
        # print(f"could not load {midi_file} because {e}")
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

    tick = 60 / (bpm * pm.resolution)
    pm._tick_scales.append((0, tick))
    pm._update_tick_to_time(0)

    pm.instruments.append(instrument)

    #######################################################
    # Function to quantize a time value to the nearest grid line
    # 1/8 note grid
    # def quantize_time(time, grid_size=1/32):
    #     grid_line = grid_size * round(time / grid_size)
    #     return grid_line
    #######################################################

    # Calculate ticks per second
    # ticks_per_second = (bpm * pm.resolution) / 60
    # def ticks_to_seconds(ticks):
    #     return ticks / ticks_per_second
    # _, start, _, _ = decode_note(notes[1])
    # offset = start

    for i, encoded_note in enumerate(notes):
        if i == 0:
            # Skip header
            continue
        pitch, start, end, velocity = decode_note(encoded_note)
        note = pretty_midi.Note(
            pitch=pitch,
            # start=quantize_time(start),   # should be in seconds not ticks
            # end=quantize_time(end),       # should be in seconds not ticks
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
