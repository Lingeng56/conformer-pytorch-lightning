import sys

import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import asyncio
import websockets
import json
import threading
from six.moves import queue
from src.deploy import build_model, preprocess_stream

model = build_model()
model.eval()

print('Model Loaded')

IP = '0.0.0.0'
PORT = 8000


class Transcoder(object):
    """
    Converts audio chunks to text
    """

    def __init__(self):
        self.buff = queue.Queue()
        self.closed = True
        self.transcript = []

    def init(self):
        model.model.init_state()
        self.buff = queue.Queue()
        self.closed = True
        # self.transcript = []

    def start(self):
        """Start up streaming speech call"""
        threading.Thread(target=self.stream_recognize).start()


    def stream_recognize(self):
        while True:
            chunk_path = self.buff.get()
            if chunk_path is None:
                continue

            chunk_inputs, chunk_inputs_len = preprocess_stream(chunk_path)
            hyps = model.model.greedy_search_streaming_app(
                chunk_inputs)
            print(hyps)
            self.transcript = hyps

    def write(self, data):
        """
        Writes data to the buffer
        """
        self.buff.put(data)


async def audio_processor(websocket):
    """
    Collects audio from the stream, writes it to buffer and return the output of Google speech to text
    """
    transcoder = Transcoder()
    transcoder.start()
    while True:
        try:
            data = await websocket.recv()
        except websockets.ConnectionClosed:
            continue
        try:
            print('Transcoder init')
            request = json.loads(data)
            if request['signal'] == 1:
                transcoder.closed = False
            elif request['signal'] == 0:
                transcoder.init()
            continue
        except Exception:
            transcoder.write(data)
            # transcoder.stream_recognize()

        # content = []
        # for w in transcoder.transcript:
        #     if w == model.model.eos:
        #         break
        #     content.append(model.char_dict[w])
        #
        # content.append(model.sp.decode(content))
        # print(' '.join(content))
        await websocket.send(transcoder.transcript)
        print(transcoder.transcript)


start_server = websockets.serve(audio_processor, IP, PORT)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
