# SD-CN-Animation
This script allows you to automate video stylization task using StableDiffusion and ControlNet. It uses a simple optical flow estimation algorithm to keep the animation stable and create an inpainting mask that is used to generate the next frame. Here is an example of a video made with this script:

[![IMAGE_ALT](https://img.youtube.com/vi/j-0niEMm6DU/0.jpg)](https://youtu.be/j-0niEMm6DU)

This script can also be using to swap the person in the video like in this example: https://youtube.com/shorts/be93_dIeZWU

## Dependencies
To install all the necessary dependencies, run this command:
```
pip install opencv-python opencv-contrib-python numpy tqdm h5py scikit-image
```
You have to set up the RAFT repository as it described here: https://github.com/princeton-vl/RAFT . Basically it just comes down to running "./download_models.sh" in RAFT folder to download the models.

## Running the scripts
This script works on top of [Automatic1111/web-ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) interface via API. To run this script you have to set it up first. You should also have[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension installed. You need to have control_hed-fp16 model installed. If you have web-ui with ControlNet working correctly, you have to also allow API to work with controlNet. To do so, go to the web-ui settings -> ControlNet tab -> Set "Allow other script to control this extension" checkbox to active and set "Multi ControlNet: Max models amount (requires restart)" to more then 2 -> press "Apply settings".

### Video To Video
#### Step 1.
To process the video, first of all you would need to precompute optical flow data before running web-ui with this command:
```
python3 compute_flow.py -i "path to your video" -o "path to output file with *.h5 format" -v -W width_of_the_flow_map -H height_of_the_flow_map
```
The main reason to do this step separately is to save precious GPU memory that will be useful to generate better quality images. Choose W and H parameters as high as your GPU can handle with respect to proportion of original video resolution. Do not worry if it higher or less then the processing resolution, flow maps will be scaled accordingly at the processing stage. This will generate quite a large file that may take up to a several gigabytes on the drive even for minute long video. If you want to process a long video consider splitting it into several parts beforehand. 

#### Step 2.
Run web-ui with '--api' flag. It is also better to use '--xformers' flag, as you would need to have the highest resolution possible and using xformers memory optimization will greatly help.   
```
bash webui.sh --xformers --api
```

#### Step 3.
Go to the **vid2vid.py** file and change main parameters (INPUT_VIDEO, FLOW_MAPS, OUTPUT_VIDEO, PROMPT, N_PROMPT, W, H) to the ones you need for your project. FLOW_MAPS parameter should contain a path to the flow file that you generated at the first step. The script is pretty simple so you may change other parameters as well, although I would recommend to leave them as is for the first time. Finally run the script with the command:
```
python3 vid2vid.py
```

### Text To Video (prototype)
This method is still in development and works on top of ‘Stable Diffusion’ and 'FloweR' - optical flow reconstruction method that is also in a yearly development stage. Do not expect much from it as it is more of a proof of a concept rather than a complete solution. All examples you can see here are originally generated at 512x512 resolution using the 'sd-v1-5-inpainting' model as a base. They were downsized and compressed for better loading speed. You can see them in their original quality in the 'examples' folder. Actual prompts used were stated in the following format "RAW photo, {subject}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3", only the 'subject' part is described in the table below.

#### Step 1.
Download 'FloweR_0.1.pth' model from here: [google disc link] and place it in the 'FloweR' folder. 

#### Step 2.
Same as with vid2vid case, run web-ui with '--api' flag. It is also better to use '--xformers' flag, as you would need to have the highest resolution possible and using xformers memory optimization will greatly help.   
```
bash webui.sh --xformers --api
```

#### Step 3.
Go to the **txt2vid.py** file and change main parameters (OUTPUT_VIDEO, PROMPT, N_PROMPT, W, H) to the ones you need for your project. Again, the script is simple so you may change other parameters if you want to. Finally run the script with the command:
```
python3 txt2vid.py
```

#### Examples:
</table>
<table class="center">
<tr>
  <td><img src="examples/flower_1.gif" raw=true></td>
  <td><img src="examples/bonfire_1.gif" raw=true></td>
  <td><img src="examples/diamond_4.gif" raw=true></td>
</tr>
<tr>
  <td width=33% align="center">"close up of a flower"</td>
  <td width=33% align="center">"bonfire near the camp in the mountains at night"</td>
  <td width=33% align="center">"close up of a diamond laying on the table"</td>
</tr>
<tr>
  <td><img src="examples/macaroni_1.gif" raw=true></td>
  <td><img src="examples/gold_1.gif" raw=true></td>
  <td><img src="examples/tree_2.gif" raw=true></td>
</tr>
<tr>
  <td width=33% align="center">"close up of macaroni on the plate"</td>
  <td width=33% align="center">"close up of golden sphere"</td>
  <td width=33% align="center">"a tree standing in the winter forest"</td>
</tr>
</table>

<!--
## Last version changes: v0.4
* Fixed issue with extreme blur accumulating at the static parts of the video.
* The order of processing was changed to achieve the best quality at different domains.
* Optical flow computation isolated into a separate script for better GPU memory management. Check out the instruction for a new processing pipeline.
-->

## Last version changes: v0.5
* Fixed an issue with wrong direction of an optical flow applied to an image.
* Added text to video mode within txt2vid.py script. Make sure to update new dependencies for this script to work!
* Added a threshold for an optical flow before processing the frame to remove white noise that might appear, as it was suggested by @alexfredo.

<!--
## Potential improvements
There are several ways overall quality of animation may be improved:
* You may use a separate processing for each camera position to get a more consistent style of the characters and less ghosting.
* Because the quality of the video depends on how good optical flow was estimated it might be beneficial to use high frame rate video as a source, so it would be easier to guess the flow properly.
* The quality of flow estimation might be greatly improved with proper flow estimation model like this one: https://github.com/autonomousvision/unimatch .
-->
## Licence
This repository can only be used for personal/research/non-commercial purposes. However, for commercial requests, please contact me directly at borsky.alexey@gmail.com
