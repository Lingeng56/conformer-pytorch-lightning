# 实时语音识别接口
import time
import numpy as np
import wave
import websocket
import threading
import asyncio

import torchaudio
SLICE_SIZE = 249040


class CG_Client:
    def __init__(self, url):
        self.ws = websocket.WebSocketApp(url,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.on_open = self.on_open

    def on_message(self, ws, message):
        print(message)

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def send_audio(self, ws):
        # First send the json string
        req = '{"platform":"wenet", "signal": 1}'
        ws.send(req)

        audio_path = "test.wav"
        # wav, sr = torchaudio.load(audio_path)
        # print(wav.shape)

        ws.send(audio_path)
        # with open(audio_path, 'rb') as wf:
        #     chunk = wf.read(SLICE_SIZE)
        #     while chunk:
        #         ws.send(chunk)
        #         time.sleep(0.02)


        req = '{"platform":"wenet", "signal": 0}'
        ws.send(req)
        time.sleep(5)
        ws.close()

    def on_open(self, ws):
        threading.Thread(target=self.send_audio, args=(ws,)).start()

    def run_forever(self):
        self.ws.run_forever()

if __name__ == '__main__':
    client = CG_Client('ws://0.0.0.0:8000')
    client.run_forever()