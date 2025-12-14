import os, sys
import math
sys.path.append(os.getcwd())

import numpy as np
from einops import rearrange
import cv2
import decord
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
import jieba
from pypinyin import Style, lazy_pinyin

import warnings
# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def align_floor_to(value, alignment):
    return int(math.floor(value / alignment) * alignment)

def align_ceil_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)

 
class EvalDataset(Dataset):
    def __init__(
        self, 
        meta_file,              
        sampling_rate, 
        **kwargs,
    ):
        super().__init__()
        self.latent_sr = sampling_rate
        self.meta_files = []
        csv_data = pd.read_csv(meta_file)
        csv_data = csv_data.fillna("")
        for idx in range(len(csv_data)):
            meta_info = {
                "data_id": str(csv_data["data_id"][idx]), 
                "ref_image_path": str(csv_data["ref_image_path"][idx]), 
                "ref_audio_path": str(csv_data["ref_audio_path"][idx]), 
                "ref_speech_content": str(csv_data["ref_speech_content"][idx]), 
                "video_path": str(csv_data["video_path"][idx]), 
                "audio_path": str(csv_data["audio_path"][idx]), 
                "speech_content": str(csv_data['speech_content'][idx]), 
                "prompt": str(csv_data['prompt'][idx]), 
                "lang": str(csv_data['lang'][idx]), 
            }
            self.meta_files.append(meta_info)
        self.meta_files = self.meta_files
        self.max_volume = kwargs.get("max_volume", 0.967926)
    
    def __len__(self):
        return len(self.meta_files)
    
    def convert_char_to_pinyin(self, text_list, polyphone=True):
        if jieba.dt.initialized is False:
            jieba.default_logger.setLevel(50)  
            jieba.initialize()

        final_text_list = []
        custom_trans = str.maketrans(
            {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
        )  

        def is_chinese(c):
            return (
                "\u3100" <= c <= "\u9fff" 
            )

        for text in text_list:
            char_list = []
            text = text.translate(custom_trans)
            for seg in jieba.cut(text):
                seg_byte_len = len(bytes(seg, "UTF-8"))
                if seg_byte_len == len(seg):  # if pure alphabets and symbols
                    if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                        char_list.append(" ")
                    char_list.extend(seg)
                elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                    seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                    for i, c in enumerate(seg):
                        if is_chinese(c):
                            char_list.append(" ")
                        char_list.append(seg_[i])
                else:  # if mixed characters, alphabets and symbols
                    for c in seg:
                        if ord(c) < 256:
                            char_list.extend(c)
                        elif is_chinese(c):
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:
                            char_list.append(c)
            final_text_list.append(char_list)

        return final_text_list
    
    def rescale_audio_volume(self, audio_input):
        max_volume = np.max(np.abs(audio_input))
        scale = self.max_volume / max_volume
        audio_input *= float(scale)
        return audio_input

    def process_frames(self, frames):
        height, width = frames.shape[1], frames.shape[2]
        min_resolution = 720
        min_resolution = align_ceil_to(min_resolution, 32)
        
        if height < width:
            scale = height / min_resolution
        else:
            scale = width / min_resolution
        resize_height = align_ceil_to(height / scale, 32)
        resize_width = align_ceil_to(width / scale, 32)
        
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((resize_height, resize_width)),              
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        
        video_frames = []
        for frame in frames:
            processed_frame = transform(frame)
            video_frames.append(processed_frame)
        
        video_frames = torch.stack(video_frames)    # (t c h w)
        video_frames = rearrange(video_frames, "t c h w -> c t h w")
        
        return video_frames

    def load_video_data(self, video_path):
        video = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(video)
        if total_frames > 81:
            raise ValueError(f"total_frames should be less than 81")
        frames = video.get_batch(range(total_frames)).asnumpy()
        return frames

    def load_image_data(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"wrong image path")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  
        img = img_rgb.astype(np.uint8)  

        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"image dimension should be 3")
        
        return img[None, ...]
    
    def __getitem__(self, idx):
        meta_data = self.meta_files[idx]

        # Load cond audio (a2v)
        if meta_data["audio_path"] != "":
            cond_audio, sampling_rate = librosa.load(meta_data["audio_path"], sr=self.latent_sr)
            cond_audio = self.rescale_audio_volume(cond_audio)
            duration = librosa.get_duration(y=cond_audio, sr=sampling_rate)
            if duration > 5.0625: 
                raise ValueError("cond audio duration should less than 5.0625s")
        else: 
            cond_audio = None
        
        # Load ref audio
        if meta_data["ref_audio_path"] != "":
            ref_audio, sampling_rate = librosa.load(meta_data["ref_audio_path"], sr=self.latent_sr)
            ref_audio = self.rescale_audio_volume(ref_audio)
            zero_audio = np.zeros((int(0.3*self.latent_sr)))
            ref_audio = np.concatenate((ref_audio, zero_audio), axis=0)
        else: 
            ref_audio = None

        # Load image 
        if meta_data["ref_image_path"] != "":
            ref_img = self.load_image_data(meta_data["ref_image_path"])
            ref_img = self.process_frames(ref_img)
        else: 
            ref_img = None

        # Load cond video (v2a)
        if meta_data["video_path"] != "":
            cond_video = self.load_video_data(meta_data["video_path"])
            cond_video = self.process_frames(cond_video)
        else: 
            cond_video = None
        
        text_char = meta_data["speech_content"]
        if text_char != "" and meta_data["lang"] == "zh":
            text_char = self.convert_char_to_pinyin([text_char])[0]
        
        ref_char = meta_data["ref_speech_content"]
        if ref_char != "" and meta_data["lang"] == "zh":
            ref_char = self.convert_char_to_pinyin([ref_char])[0]
        
        if ref_char != "":
            if meta_data["lang"] == "zh":
                cat_char = ref_char + [" "] + text_char
            else: 
                cat_char = ref_char + " " + text_char
        else: 
            cat_char = text_char
        
                
        prompt = meta_data["prompt"]
            
        data_id = meta_data["data_id"]
            
        batch = {
            "text_char": text_char, 
            "ref_text_char": ref_char, 
            "cat_text_char": cat_char, 
            "data_id": data_id,                     
            "cond_audio": torch.from_numpy(cond_audio) if cond_audio is not None else None, 
            "ref_audio": torch.from_numpy(ref_audio) if ref_audio is not None else None, 
            "sample_rate": self.latent_sr, 
            "text_prompt": prompt, 
            "ref_img": ref_img,          
            "cond_video": cond_video, 
            "lang": meta_data["lang"]         
        }
        
        return batch


def collate_fn(batch):
    # import pdb; pdb.set_trace()
    list_batch = {key: [] for key in batch[0].keys()}
    for cur_batch in batch:
        for key, val in cur_batch.items():
            list_batch[key].append(val)
    
    return list_batch
def build_vocab_mapper():
    tokenizer_path = "vocab.txt"
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
    return vocab_char_map, vocab_size

def list_str_to_idx(text, vocab_char_map, padding_value=-1):
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    list_len = [len(c) for c in list_idx_tensors]
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text, list_len

if __name__ == "__main__":
    pass
    