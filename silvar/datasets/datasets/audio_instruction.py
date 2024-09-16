import os
import json
import random
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

class AudioInstruction(Dataset):
    def __init__(self, vis_processor, text_processor, audio_processor, audio_dir, vis_root, ann_path, prompt_test=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        audio_dir (string): Root direction of audio
        """
        self.vis_root = vis_root
        self.audio_dir = audio_dir

        self.vis_processor = vis_processor

        self.text_processor = text_processor
        self.audio_processor = audio_processor

        # if prompt_test is None:
        #     self.instruction_pool = [
        #         "[grounding] please describe this image in details with radiological features. Use two sentences unless there are no findings. The first sentence should list the global diseases present in the image, and the second should list local diseases with localized bounding boxes."
        #         ]
        # else:
        #     self.instruction_pool = [prompt_test]

        # path: ../data/LISA/json/train.json
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['image']
        image_id, _ = os.path.splitext(info['image'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        answer = info['outputs']
        # instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img>" 

        # audio
        audio_file, _ = os.path.splitext(info['image'])
        audio_path = os.path.join(self.audio_dir, f'{audio_file}.wav')
        waveform, sample_rate = torchaudio.load(audio_path)
        # waveform, sample_rate, audio_file = random.choice(self.audio_data) # sample_rate = 24000

        # print('shape of audio before:', waveform.shape, sample_rate)

        waveform_array = waveform.squeeze().numpy()

        waveform = self.audio_processor(waveform_array) #, sampling_rate=16000, return_tensors="pt").input_features
        waveform = waveform.squeeze()

        return {
            "image": image,
            "audio": waveform, #torch.rand(80, 3000, dtype=torch.float16), # double length of the max_source_positions
            "instruction_input": instruction,
            "answer": answer,
            "image_id": image_id,
        }
