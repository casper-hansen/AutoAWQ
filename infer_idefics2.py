import requests
import torch
from PIL import Image

from awq import AutoAWQForCausalLM
from transformers import AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format


quant_path = "/admin/home/victor/code/idefics2-awq"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, safetensors=True, device_map={"": 0})
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2")

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)

image_seq_len = model.config.perceiver_config.resampler_n_latents
BOS_TOKEN = processor.tokenizer.bos_token
BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


# The processor is the same as the Idefics processor except for the BILINEAR interpolation,
# so this is a hack in order to redefine ONLY the transform method
def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)

    height, width = x.shape[:2]
    aspect_ratio = width / height
    if width >= height and width > 980:
        width = 980
        height = int(width / aspect_ratio)
    elif height > width and height > 980:
        height = 980
        width = int(height * aspect_ratio)
    width = max(width, 378)
    height = max(height, 378)

    x = resize(x, (height, width), resample=PILImageResampling.BILINEAR)
    x = processor.image_processor.rescale(x, scale=1 / 255)
    x = processor.image_processor.normalize(
        x,
        mean=processor.image_processor.image_mean,
        std=processor.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x


inputs = processor.tokenizer(
    f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>In this image, we see",
    return_tensors="pt",
    add_special_tokens=False,
)
inputs["pixel_values"] = processor.image_processor([raw_image], transform=custom_transform)
inputs = {
    k: v.to(0, torch.float16) if k != "input_ids" else v.to(0)
    for k, v in inputs.items()
}

# Generate output
generation_output = model.generate(
    **inputs,
    max_new_tokens=512,
    bad_words_ids=BAD_WORDS_IDS
)

print(processor.decode(generation_output[0], skip_special_tokens=True))