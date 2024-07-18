import pyttsx3

# 初始化 pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 130)
# 将文本转换为语音
text = "秋美琴你好，請到指定位子"
engine.save_to_file(text, './static/'+str(1)+'.wav')

# 执行文本到语音转换并保存到文件
engine.runAndWait()