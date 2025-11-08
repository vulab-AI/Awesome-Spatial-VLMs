# Finetune LLaVA-3D on Custom Datasets

## Data Preparation

We recommend to organize you own data in the [EmbodiedScan](https://raw.githubusercontent.com/OpenRobotLab/EmbodiedScan/refs/heads/main/data/README.md) format.

For each 3D scene, we need RGB-D pairs of the same size, where RGB is the color image and D is the depth image.

```
data
├── scannet
│   ├── <scene_id>
│   ├── 000000.jpg  # RGB image
│   ├── 000000.png  # Depth image
│   ├── 000010.jpg 
│   ├── 000010.png
│   ├── ...
```

After organizing your own data, you need to generate a info file which contain the 3D scene and camera paramters inforamtion in the following format. Each 3D scene is a dict containing the information:

1. Top Level Structure:

```python
{
    "scannet/scene0191_00": {               # Scene folder
        "scannet/scene0191_00/...": {...},  # Image data
        ...,
        "axis_align_matrix": [...],         # Axis_align_matrix (optional): 4x4 matrix for aligning the scene to a canonical coordinate system
        "intrinsic": [...]                  # 4x4 Camera intrinsic parameters.
    }
}
```

2. For Each Image in the Scene:

```python
"scannet/scene0191_00/00000.jpg": {
    "pose": [                   # 4x4 camera pose matrix
        [0.246516, -0.470365, 0.847341, 3.043058],  # Row 1: rotation + translation
        [-0.959136, 0.006886, 0.282862, 2.955299],  # Row 2: rotation + translation
        [-0.138884, -0.882445, -0.449446, 1.551102], # Row 3: rotation + translation
        [0.0, 0.0, 0.0, 1.0]                        # Row 4: homogeneous coordinates
    ],
    "depth": "scannet/posed_images/scene0191_00/00000.png"  # Path to depth map
}
```

Please refer to our provided [Camera Parameters File](https://drive.google.com/file/d/1a-1MCFLkfoXNgn9XdlmS9Gnzplrzw7vf/view?usp=drive_link) for more details.


## Dataset Format

Since our model supports 2D image data and 3D data (Posed RGB-D image data):

* For **2D data format**: Please refer to the [LLaVA documentation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)

* For **3D data format**: Convert your data into a JSON file containing a list of samples. Each sample should include:
  - `id`: A unique identifier for the sample
  - `video`: The file path to the 3D scene
  - `conversations`: The human-AI conversation data for the scene

Here we further introduce the data format for different tasks.

### Pure-text 3D QA / Region and Scene Captioning 

When specifying your data path in LLaVA-3D, the `video` field can point to either a single 3D scene path or a directory containing multiple images of the 3D scene. The model distinguishes between 3D and 2D data based on the field name in your sample dictionary - using `video` for 3D data and `image` for 2D data. Additionally, when writing questions about 3D scenes, you must include the `<video>` tag to properly reference the scene. We provide two examples that illustrate these concepts:

```json
# Single 3D scene path
{
"id": "0",
"video": "scannet/scene0011_00",
"conversations": [
    {
    "from": "human",
    "value": "<video>\nWhat color is the chair in the kitchen? Answer the question using one word or one phrase."
    },
    {
    "from": "gpt",
    "value": "Dark brown."
    },
]
}

# Multiple images of the 3D scene
{
    "id": 2,
    "video": [
        "scannet/posed_images/scene0678_00/01000.jpg",
        "scannet/posed_images/scene0678_00/01470.jpg",
        "scannet/posed_images/scene0678_00/01380.jpg",
        "scannet/posed_images/scene0678_00/00550.jpg",
        "scannet/posed_images/scene0678_00/00760.jpg",
        "scannet/posed_images/scene0678_00/01210.jpg",
        "scannet/posed_images/scene0678_00/00730.jpg",
        "scannet/posed_images/scene0678_00/00290.jpg",
        "scannet/posed_images/scene0678_00/01670.jpg",
        "scannet/posed_images/scene0678_00/01610.jpg"
    ],
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nDescribe the scene concisely."
        },
        {
            "from": "gpt",
            "value": "The area seems to be a communal laundry room, typically found in apartment buildings or dormitories, specifically designed for residents to wash and dry their clothes. The room layout is reasonable, with a row of washers and dryers placed together, providing ample space for multiple users. Though no specific dimensions are given, the space appears sufficient for its intended purpose. The room features a standard entry door, multiple windows for ventilation, fixed walls for hanging pictures and notices, tile flooring for easy cleaning, and possibly a simple acoustic tile ceiling. Soft furnishings include baskets and trash cans for laundry organization and storage. Decorative details seem minimal, but there are some pictures on the walls to enhance the atmosphere. The light switch at the entrance indicates an overhead light, which is very practical for the functional space, ensuring users have good visibility at all times. The color scheme of the room is neutral, possibly to match the functional aesthetics of the public space. The style is practical, focusing on utility rather than decoration. A special feature is the table next to the window, used for storage or as a folding area. The fan next to the window indicates an emphasis on ventilation."
        }
    ]
}
```


### Objected-guied 3D QA / Dense Object Captioning

For interactions with 3D scenes, some instructions may require spatial information specified through coordinates. These coordinates are stored in the `boxes` metadata field. For 3D objects interaction, we recommend using their center coordinates. Our model supports coordinates for both single and multiple objects as input. When training your own model, these coordinates aren't limited to objects - they can represent any position in 3D space according to your task.

In the instruction prompt, we use the `<boxes>` special token as a placeholder, which the model will replace with the appropriate 3D coordinate tokens during processing.

```python
# for single object interaction
    {
        "id": 3,
        "prompt_id": "1mp3d_0047_region0_11",
        "video": "matterport3d/1mp3d_0047_region0",
        "target": {
            "boxes": [
                [
                    -2.291,
                    -1.915,
                    1.114
                ]
            ]
        },
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nThe related object is located at <boxes>. Describe the object in detail.",
                "boxes_seq": [
                    [
                        0
                    ]
                ]
            },
            {
                "from": "gpt",
                "value": "The door is a vertical, slender door with a stylish and simple design. It is painted in a light grey color, matching the surrounding walls. There is a minimalist-style handle on the door, also grey in color and made of metal. The door is open. In terms of size, it is of average height and a moderate size, suitable for a typical room entrance. This door's design is simple and practical, perfect for everyday use, serving as the entrance to another space."
            }
        ]
    }

# for multiple objects interaction
    {
        "id": 4,
        "prompt_id": "QA_Inter_Attribute_OO__scene0274_02__0",
        "video": "scannet/scene0274_02",
        "target": {
            "boxes": [
                [
                    -0.028,
                    -0.285,
                    0.516,
                ],
                [
                    -0.611,
                    2.014,
                    0.787
                ]
            ]
        },
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nThe related objects are located at <boxes>. Are these two objects same in coarse grained category? why?",
                "boxes_seq": [
                    [
                        0,
                        1
                    ]
                ]
            },
            {
                "from": "gpt",
                "value": [
                    "Yes, the chair and the table are both furniture."
                ]
            }
        ]
    }
```

For 3D Dense Object Captioning tasks, some datasets require not only object descriptions but also their corresponding 3D Bounding Boxes as output. For such task, we only need to output the 3D Bounding Boxes in text format.

```json
    {
        "id": 0,
        "video": "scannet/scene0000_00",
        "target": {
            "boxes": [
                [
                    -3.150,
                    -2.905,
                    0.940
                ]
            ]
        },
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nThe related object is located at <boxes>. Describe the object first: begin by detailing the visual aspects of this object before delving into its spatial context among other elements within the scene, and then localize the object in the 3D scene.",
                "boxes_seq": [
                    [
                        0
                    ]
                ]
            },
            {
                "from": "gpt",
                "value": "A white cabinet in the corner of the room. In the direction from the door and from the inside . It will be on the left, there is a small brown table on the left side of the cabinet and a smaller table on the right side of the cabinet It is localized at [-3.150, -2.905, 0.940, 0.601, 1.218, 1.938]"
            }
        ]
    }
```

### 3D Visual Grounding

For 3D visual grounding tasks, outputting 3D bounding boxes directly in text format is challenging. Instead, we use a special <LOC> token that works with our grounding decoder. We will provide detailed documentation for this feature when we release the grounding-related code.


## Command

You could finetune from our provided LLaVA-3D checkpoints with full-model finetuning following this [script](hhttps://github.com/ZCMax/LLaVA-3D/blob/main/scripts/train/finetune.sh). You may need to adjust the hyperparameters to fit each specific dataset and your hardware constraint.


