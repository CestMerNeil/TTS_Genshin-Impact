import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
    output_path = "/Users/neil/Code/TTS_GI/outputs"

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path="/Users/neil/Code/TTS_GI/dataset",
    )

    audio_config = VitsAudioConfig(
        sample_rate=16000,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="VITS_Kamisato_Ayaka",
        run_description="VITS Kamisato Ayaka / Genshin Impact",
        batch_size=16,
        eval_batch_size=16,
        batch_group_size=1,
        num_loader_workers=8,
        num_eval_loader_workers=8,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        text_cleaner="chinese_mandarin_cleaners",
        use_phonemes=True,
        phoneme_language="zh-cn",
        phoneme_cache_path=None,
        compute_input_seq_cache=True,
        print_step=1,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        cudnn_enable=False,
        use_noise_augment=True,
        test_sentences=[
            ["你好啊，小唐，你今天过的开心么？"],
            ["好久没有联系啦，你最近还好吗？"],
            ["我最近在学习新的技术，感觉很有趣。"],
        ],
    )

    print(config)

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)


    train_samples, eval_samples = load_tts_samples(
        dataset_config, 
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(
        config, 
        ap, 
        tokenizer,
        speaker_manager=None,
    )

    print("===================================")
    print("Model:")
    print(model)

    trainer = Trainer(
        TrainerArgs(),
        config, 
        output_path,
        model=model, 
        train_samples=train_samples,
        eval_samples=eval_samples,
        
    )

    print("===================================")
    print("Trainer:")
    print(trainer)

    trainer.fit()

if __name__ == "__main__":
    main()