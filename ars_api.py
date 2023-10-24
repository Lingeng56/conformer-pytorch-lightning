# 录音文件识别极速版接口
import requests


url = "http://demo.dengfeng.fun:28083/recognize/"
api_key = "demo.dengfeng.fun"
audio = open("samples/t14.wav", "rb")  # 修改为你的音频路径
content = {}
files = {"audio": audio}
response = requests.post(url, data=content, files=files)
result = response.json()
if result["status"] == "success" or result["status"] == "SUCCESS":
    print("识别状态：", result["status"], '识别结果：', result["message"])
else:
    print("发生错误：", result["status"], '识别结果：', result["message"])
