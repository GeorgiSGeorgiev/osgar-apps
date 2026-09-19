"""
   Matty audio - after dtc-systems/dtc_audio.py

   Robotour rules (v11, "Reaching goal"): the robot has to indicate when the
   goal is reached, e.g. by a sound, in both the loading and the unloading
   zone - and homologation tests it. This node listens to the router's
   route_hint and plays sounds/<name>.mp3 (or .wav) on

     waiting   the router waits for a QR command - at boot, after the
               emergency stop is released, after a "stop" QR
     arrived   the router reports the target reached; repeated so a
               referee walking up still hears it
     target    a route was planned for a new QR target (route_plan) -
               silent unless target_sound is set

   Per event: <event>_sound (null or "" = silent), <event>_repeat_sec and
   <event>_repeat_count. A named file that is missing falls back to a
   generated beep, so the robot is never silent for want of a recording.

   Playback is ffplay as in dtc_audio.py, on the default audio output (the
   AUX jack or a Bluetooth speaker, whichever is the default). Each sound
   blocks this node's own thread only, with a timeout, and a missing ffplay
   or speaker only prints: nothing here can stop the robot.
"""
import os
from subprocess import call, TimeoutExpired

from osgar.node import Node

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')

# used when the named file is missing - ffmpeg's own tone generator
BEEPS = {
    'waiting': "aevalsrc='0.7*sin(2*PI*660*t)*lt(mod(t,0.5),0.3)':d=1.0",   # two beeps
    'arrived': "aevalsrc='0.8*sin(2*PI*880*t)*lt(mod(t,0.4),0.25)':d=1.6",  # four beeps
    'target': "aevalsrc='0.8*sin(2*PI*1320*t)':d=0.3",                    # one short beep
    'keepalive': "aevalsrc='0.02*sin(2*PI*200*t)':d=0.5",                 # barely audible
}
DEFAULTS = {             # sound, repeat_sec, repeat_count
    'waiting': ('waiting', 30.0, 1),
    'arrived': ('arrived', 10.0, 3),
    'target': (None, 0.0, 1),
}


class MattyAudio(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('playing')
        self.sounds_dir = config.get('sounds_dir', SOUNDS_DIR)
        self.sound, self.repeat_sec, self.repeat_count = {}, {}, {}
        for event, (sound, sec, count) in DEFAULTS.items():
            self.sound[event] = config.get(event + '_sound', sound)
            self.repeat_sec[event] = config.get(event + '_repeat_sec', sec)
            self.repeat_count[event] = config.get(event + '_repeat_count', count)
        # Many speakers switch themselves off after some minutes of silence,
        # which would make the arrival silent. >0 plays a barely audible
        # keepalive that often.
        self.keepalive_sec = config.get('keepalive_sec', 0.0)
        self.sound['keepalive'] = config.get('keepalive_sound', 'keepalive')
        self.timeout_sec = config.get('timeout_sec', 15.0)
        self._event = None
        self._count = 0
        self._last_event_play = None
        self._last_play = None
        self._problems = set()

    def _command(self, kind):
        source = ['-f', 'lavfi', BEEPS[kind]]
        for ext in ('.mp3', '.wav'):
            path = os.path.join(self.sounds_dir, self.sound[kind] + ext)
            if os.path.isfile(path):
                source = [path]
                break
        return ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'error'] + source

    def _play(self, kind, why):
        cmd = self._command(kind)
        self.publish('playing', [kind, True])
        if kind != 'keepalive':
            print(self.time, 'AUDIO: %s - %s' % (why, cmd[-1]))
        try:
            call(cmd, timeout=self.timeout_sec)
        except TimeoutExpired:
            print(self.time, 'AUDIO: %s still playing after %.0fs - stopped' % (kind, self.timeout_sec))
        except OSError as e:
            if kind not in self._problems:      # e.g. ffplay not installed - say it once
                self._problems.add(kind)
                print(self.time, 'AUDIO: cannot play %s: %s' % (kind, e))
        self._last_play = self.time
        self.publish('playing', [kind, False])

    def _since(self, t):
        return float('inf') if t is None or self.time is None else (self.time - t).total_seconds()

    def on_route_hint(self, data):
        if data.get('arrived'):
            event = 'arrived'
        elif data.get('state') == 'waiting':
            event = 'waiting'
        else:
            event = None
        if event != self._event:
            self._event, self._count, self._last_event_play = event, 0, None
        if (event is not None and self.sound[event]
                and self._count < self.repeat_count[event]
                and self._since(self._last_event_play) >= self.repeat_sec[event]):
            self._count += 1
            self._last_event_play = self.time
            self._play(event, '%s (%d/%d)' % (event, self._count, self.repeat_count[event]))
        elif self.keepalive_sec > 0 and self._since(self._last_play) >= self.keepalive_sec:
            self._play('keepalive', 'keepalive')

    def on_route_plan(self, data):
        if data.get('reason') == 'new-target' and self.sound['target']:
            self._play('target', 'new target accepted, %.0f m' % data.get('total_m', 0))

#----------------------------------------------

def self_test(config):
    """Plays every configured sound once through the real speaker."""
    from unittest.mock import MagicMock
    audio_player = MattyAudio(bus=MagicMock(), config=config)
    for kind in ('waiting', 'arrived', 'target'):
        if audio_player.sound[kind]:
            audio_player._play(kind, 'self test')


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Matty Audio - plays each configured sound once')
    parser.add_argument('--config', help='robot config to take audio_player.init from')
    args = parser.parse_args()
    init = {}
    if args.config:
        with open(args.config, encoding='utf-8') as f:
            init = json.load(f)['robot']['modules']['audio_player']['init']
    self_test(init)

# vim: expandtab sw=4 ts=4
