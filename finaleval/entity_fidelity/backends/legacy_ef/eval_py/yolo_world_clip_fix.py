_base_ = "yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py"

text_model_name = "openai/clip-vit-base-patch32"

model = dict(
    vision_model=text_model_name,
    backbone=dict(
        text_model=dict(
            model_name=text_model_name,
        ),
    ),
)
