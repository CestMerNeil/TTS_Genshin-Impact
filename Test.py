from TTS.utils.synthesizer import Synthesizer
import soundfile as sf

model_path = "best_model.pth"
config_path = "config.json"

synthesizer =  Synthesizer(model_path, config_path)

text = "今天也是喜欢小唐的一天奥。"

wav = synthesizer.tts(text)
sf.write("output.wav", wav, 16000)