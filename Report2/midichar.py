# import logging
import pretty_midi
import numpy as np
import logging
from typing import Any, Tuple


def encode_note(note: Any) -> int:
    """
    | VELOCITY  | STEP      |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |

    | DURATION  | PITCH     |
    | --------- | --------- |
    | 0000 0000 | 0000 0000 |
    """
    start = note.start
    end = note.end
    step = start
    velocity = note.velocity
    pitch = note.pitch
    duration = end # (end-start)

    int_velocity = int(velocity)
    int_step = int(255 * (step/100))
    int_duration = int(255 * (duration/100))

    print("-->", pitch, step, duration, velocity)
    # print("------->", int_step, int_duration)

    encoded_note = 0
    encoded_note += int_velocity << 24
    encoded_note += int_step << 16
    encoded_note += int_duration << 8
    encoded_note += pitch

    return encoded_note


def decode_note(encoded_note: int) -> Tuple:
    pitch = (encoded_note) & 255
    velocity = (encoded_note >> 24) & 255
    int_step = (encoded_note >> 16) & 255
    int_duration = (encoded_note >> 8) & 255
    # print(f"{encoded_note:32b}")

    f_step = 100*(int_step/255)
    f_duration = 100*(int_duration/255)

    print("-->", pitch, f_step, f_duration, velocity)

    return (pitch, f_step, f_duration, velocity)


def encode_midi(midi_file: str,
                window_size=64,
                instrument_index=None) -> np.array:
    """
    Encode a midi file into a numpy array of integers using the
    instrument index with a max window size of window_size
    (window size is basically the number of note on events)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = None

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

        notes = []
        # sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        sorted_notes = instrument.notes

        i = 0
        for note in sorted_notes:
            encoded_note = encode_note(note)
            notes.append(encoded_note)
            i += 1
            if i >= window_size:
                break
    except Exception as e:
        print(f"could not load {midi_file} because {e}")
        return None

    return np.array(notes)


def decode_midi(
    notes: np.array,
    out_file: str,
    instrument_name: str = "Standard Kit",
    bpm: int = 95
) -> pretty_midi.PrettyMIDI:
    """
    Given an array of encoded midi integers write a new midi file using
    the general midi instrument name as the instrument to use
    """

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=0, #pretty_midi.instrument_name_to_program(instrument_name),
        is_drum=True
    )

    # Add a time signature change
    time_signature = pretty_midi.TimeSignature(numerator=4, denominator=4,
                                               time=0)
    pm.time_signature_changes.append(time_signature)

    # Key number according to [0, 11] Major, [12, 23] minor. For example, 0
    # is C Major, 12 is C minor - time is when to apply the change
    key_signature = pretty_midi.KeySignature(key_number=0, time=0)
    pm.key_signature_changes.append(key_signature)

    # print("-->", 60/(bpm*pm.resolution))
    tick = 60/(bpm*pm.resolution)
    pm._tick_scales.append((0, tick))
    pm._update_tick_to_time(0)

    pm.instruments.append(instrument)

    #prev_start = 0.0
    for _, encoded_note in enumerate(notes):
        pitch, start, end, velocity = decode_note(encoded_note)
        note = pretty_midi.Note(
            pitch=pitch,
            start=start,   # should be in seconds not ticks
            end=end,       # should be in seconds not ticks
            velocity=velocity,
        )
        instrument.notes.append(note)

    pm.write(out_file)
    return pm


def encoded_notes_to_str(raw_notes: np.array) -> str:
    midi_chars = []
    for _, raw_note in enumerate(raw_notes):
        try:
            midi_char = int(raw_note.astype(np.uint32))
            byte_array = midi_char.to_bytes(4, 'big')
            unicode_character = byte_array.decode('utf-8')
            midi_chars.append(unicode_character)
        except Exception as e:
            print(e)
            # print(f"{midi_char:32b} - {byte_array}")

    midi_string = "".join(midi_chars)
    return midi_string


def str_to_encoded_notes(string: str) -> np.array:
    from_file = []
    for c in string:
        from_file.append(int.from_bytes(bytes(c, "utf-8"), 'big'))
    return np.array(from_file)
