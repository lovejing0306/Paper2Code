# coding=utf-8

import os
import random
import decord  # isort:skip
import traceback
import torch
import torchvision.transforms.functional as TTF
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

decord.bridge.set_bridge("torch")


class ImageOrVideoDataset(Dataset):
    def __init__(
        self,
        dataset_file: Optional[str],
        delimiter: str = "@@",
        frame_stride_min: int = 1,
        frame_stride_max: int = 8,
        channel_order: str = "TCHW",
        start_skip_frms_num: int = 0,
        end_skip_frms_num: int = 0,
        enable_shuffle=False,
    ) -> None:
        super().__init__()
        self.dataset_file = (
            dataset_file if isinstance(dataset_file, list) else [dataset_file]
        )
        self.delimiter = delimiter
        self.frame_stride_min = frame_stride_min
        self.frame_stride_max = frame_stride_max
        self.channel_order = channel_order
        self.start_skip_frms_num = start_skip_frms_num
        self.end_skip_frms_num = end_skip_frms_num

        self.data_paths = self._load_dataset()

        if enable_shuffle:
            self._shuffle()

        self.video_transforms = transforms.Compose(
            [
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def _load_dataset(self):
        data_paths = []
        for data_file in self.dataset_file:
            assert os.path.exists(data_file), (
                f"Dataset file {data_file} does not exist."
            )
            with open(data_file, "r", encoding="utf-8") as file:
                for line in file.readlines():
                    try:
                        line = line.strip("\n")
                        assert self.delimiter in line, (
                            f"Expected delimiter {self.delimiter} in line {line}"
                        )
                        items = line.split(self.delimiter)
                        if len(items) < 2:
                            raise ValueError(
                                f"Expected 2 items separated by {self.delimiter} in line {line}"
                            )
                        input_path, caption_path = items[0].strip(), items[1].strip()

                        data_path = []
                        if os.path.exists(input_path) and os.path.isfile(input_path):
                            data_path.append(input_path)
                        else:
                            raise ValueError(f"Video path {input_path} does not exist.")

                        if os.path.exists(caption_path) and os.path.isfile(
                            caption_path
                        ):
                            data_path.append(caption_path)
                        else:
                            raise ValueError(
                                f"Caption path {caption_path} does not exist."
                            )
                        data_paths.append(data_path)
                    except Exception as e:
                        print(e)
        return data_paths

    def _shuffle(self):
        random.shuffle(self.data_paths)

    def _preprocess_image(self, path):
        # TODO(aryan): Support alpha channel in future by whitening background
        image = Image.open(path).convert("RGB")
        image = TTF.to_tensor(image)
        image = image * 2.0 - 1.0
        image = image.unsqueeze(
            0
        ).contiguous()  # [C, H, W] -> [1, C, H, W] (1-frame video)
        return image, 1

    def _preprocess_video(self, path):
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        Returns a [F, C, H, W] video tensor.
        """
        video_reader = decord.VideoReader(uri=path)
        video_num_frames = len(video_reader)
        assert video_num_frames > (self.start_skip_frms_num + self.end_skip_frms_num)
        video_num_frames -= self.start_skip_frms_num + self.end_skip_frms_num

        sample_interval = random.randint(
            self.frame_stride_min, self.frame_stride_max
        )  # 可以采样到最大和最小值
        indices = [
            self.start_skip_frms_num + sample_interval * i
            for i in range(video_num_frames // sample_interval)
        ]
        frames = video_reader.get_batch(indices)
        frames = frames.float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)
        return frames, sample_interval

    def _preprocess_caption(self, path):
        with open(path, "r", encoding="utf-8") as file:
            caption = file.read().strip("\n")
            caption = caption.replace("\n", " ")
        return caption

    def __getitem__(self, index: int):
        while True:
            try:
                items = self.data_paths[index]
                input_path, caption_path = items
                input_path = Path(input_path)

                if input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    inputs, sample_interval = self._preprocess_image(input_path)
                elif input_path.suffix.lower() in [".mp4"]:
                    inputs, sample_interval = self._preprocess_video(input_path)
                else:
                    raise ValueError(f"Unsupported video format: {input_path.suffix}")

                caption = self._preprocess_caption(caption_path)

                break
            except Exception as e:
                print(f"Error preprocessing data for index {index}: {e}. Retrying...")
                print(traceback.format_exc())
                index = random.randint(0, len(self.data_paths) - 1)
                continue
        num_frames = inputs.shape[0]
        if self.channel_order == "CTHW":
            inputs = inputs.permute(
                1, 0, 2, 3
            ).contiguous()  # [T, C, H, W] -> [C, T, H, W]

        return {
            "sample_path": input_path,
            "prompt": caption,
            "input": inputs,
            "input_metadata": {
                "num_frames": num_frames,
                "height": inputs.shape[2],
                "width": inputs.shape[3],
                "sample_interval": sample_interval,
            },
        }

    def __len__(self) -> int:
        return len(self.data_paths)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dataset = ImageOrVideoDataset(
        dataset_file="",
        delimiter="@@",
        frame_stride_min=1,
        frame_stride_max=8,
        channel_order="TCHW",
        start_skip_frms_num=0,
        end_skip_frms_num=0,
        enable_shuffle=True,
    )

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=0, sampler=sampler, pin_memory=True
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    for batch_idx, batch in enumerate(dataloader):
        prompt = batch["prompt"]
        video = batch["input"]

        if rank == 0:
            print(prompt)
            print(video.shape)
