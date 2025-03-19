import os
import argparse

import torch
import torchvision

from PIL import Image
from diffusers import StableDiffusionGLIGENPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with text boxes.")

    def box_type(values):
        try:
            box = list(map(float, values.split(',')))
            if len(box) != 4:
                raise argparse.ArgumentTypeError(
                    f"each box must have exactly 4 elements, got: {len(box)}")
            return box
        except:
            raise argparse.ArgumentTypeError(
                f"invalid box format, expected: xmin,ymin,xmax,ymax, got: {values}")

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="longlian/igligen-sd2.1-v1.0",
        help="Pretrained model name or path."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Model revision (must be 'fp32' or 'fp16')"
    )

    # Inference
    parser.add_argument(
        "--prompt",
        type=str,
        help="The prompt or prompts to guide image generation."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."
    )
    parser.add_argument(
        "inpaint_image_path",
        type=str,
        help="The path to the image to be inpainted with text boxes."
    )

    # GLIGEN
    parser.add_argument(
        "--ground_phrases",
        nargs='*',
        type=str,
        help="The phrases to guide what to include in each of the regions defined by the corresponding "
             "`gligen_boxes`. There should only be one phrase per bounding box."
    )
    parser.add_argument(
        "--ground_boxes",
        nargs='*',
        type=box_type,
        help="The bounding boxes that identify rectangular regions of the image that are going to be filled with the "
             "content described by the corresponding `gligen_phrases`. Each rectangular box is defined as a "
             "`List[float]` of 4 elements `[xmin, ymin, xmax, ymax]` where each value is between [0,1]."
    )
    parser.add_argument(
        "--gligen_scheduled_sampling_beta",
        type=float,
        default=1.0,
        help="Scheduled Sampling factor from [GLIGEN: Open-Set Grounded Text-to-Image "
        "Generation](https: // arxiv.org/pdf/2301.07093.pdf). Scheduled Sampling factor is only varied for "
        "scheduled sampling during inference for improved quality and controllability."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gen_images",
        help="The directory where the generated images will be saved."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.revision == "fp16":
        torch_dtype = torch.float16
    elif args.revision == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported revision: {args.revision}")

    pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        torch_dtype=torch_dtype
    )
    pipeline.to("cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    # prompt = "a dog and a birthday cake"

    # ground_phrases = ["a dog", "a birthday cake"]
    # ground_boxes = [
    #     [0.1871, 0.3048, 0.4419, 0.5562],
    #     [0.2152, 0.6792, 0.7671, 0.9482]
    # ]

    images = pipeline(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        gligen_inpaint_image=Image.open(
            args.inpaint_image_path).convert('RGB'),
        gligen_phrases=args.ground_phrases,
        gligen_boxes=args.ground_boxes,
        gligen_scheduled_sampling_beta=args.gligen_scheduled_sampling_beta,
        output_type="numpy"
    ).images

    images = torch.stack([torch.from_numpy(img)
                         for img in images]).permute(0, 3, 1, 2)
    torchvision.utils.save_image(
        images,
        os.path.join(args.output_dir, "inpaint_text_box.png"),
        nrow=args.num_images_per_prompt,
        normalize=False
    )

    # python generation_text_box.py \
    #   --pretrained_model_name_or_path "longlian/igligen-sd2.1-v1.0" \
    #   --revsion "fp16" --prompt "a dog and a birthday cake" \
    #   --num_images_per_prompt 2 --num_inference_steps 50 \
    #   --inpaint_image_path "./data/inpaint_image.png" \
    #   --ground_phrases "a dog" "a birthday cake" \
    #   --ground_boxes 0.1871,0.3048,0.4419,0.5562 0.2152,0.6792,0.7671,0.9482 \
    #   --gligen_scheduled_sampling_beta 0.3 \
    #   --output_dir "./gen_images"
