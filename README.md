# RVC Webui Premium 1-Click Installers for Windows, RunPod, Massed Compute, SimplePod with All GPUs Supporting Inlcuding RTX 1000, 2000, 3000, 4000, 5000 GPUs with fastest libraries

## Get the app installer zip file from here : https://www.patreon.com/posts/149104996

Latest installer zip file : [**RVC_Installer.zip**](https://www.patreon.com/posts/149104996)

- This app was requested so I found the most up to date fork and forked and improved it
  - Using this app is not trivial so look for guides and tutorials
- I have downloaded pre-trained audio models from here and tested and works
  - [https://huggingface.co/QuickWick/Music-AI-Voices/tree/main](https://huggingface.co/QuickWick/Music-AI-Voices/tree/main) - massive archive
  - [https://huggingface.co/Coolwowsocoolwow](https://huggingface.co/Coolwowsocoolwow) - massive archive
- There are 450 GB of pre-trained voice models pth files in above repo that you can use
  - Put those pth files into `Q:\RVC_Installer_V1\RVC_Premium\assets\weights`
  - Put those index files into `G:\RVC_Installer_V1\RVC_Premium\assets\indices`
  - You will see auto downloaded models so you will understand how it works
- Our installer will auto download all of the necessary models for voice training and other tasks automatically
- Our installer will also download over 30 demo voices
- I don't have much experience with this app but you can find lots of tutorials to learn more about it and this is the most up to date version forked from here : [https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- Our installer will generate a Python 3.10 venv and install all the libraries inside it
  - Torch 2.9.1 with CUDA 13, Flash Attention, Sage Attention, xFormers, compatible with all GPUs starting from RTX 1000 to RTX 5000 GPUs
- We also have installers for RunPod, Massed Compute and SimplePod
- This is a very advanced version of original RVC Webui Premium developed by SECourses

## 10 April 2026 V4

- Full tutorial video published that shows how to install and use on Windows, RunPod, SimplePod and Massed Compute : [https://youtu.be/ZRrzvD4wNys](https://youtu.be/ZRrzvD4wNys)
- Upon request, merge voice feature implemented to generate custom mixed voices
- Supports merging more than 2 voices with rates as well and including indexes (not mandatory)
- Fairseq library installation was extremely hard on Windows therefore I hace pre-compiled it and now it is auto installed with support to all GPUs out ther
- Download latest zip file, extract and overwrite all files and then Just run `Windows_Install_and_Update.bat` file

<p align="center">
 <img height="600" alt="image" src="https://github.com/user-attachments/assets/46b01b6e-b12f-4d47-88c7-318fedfa3fa5" />
</p>

<p align="center">
 <img  height="600" alt="image" src="https://github.com/user-attachments/assets/0b04aa20-03d7-49e9-9e5c-e3e284195e2e" />
</p>

## 18 March 2026 V2

- I have upgraded the app significantly
- Now we are moved to newest Torch 2.9.1 and CUDA 13
- Workin on all GPUs, Windows, Linux, RunPod, Massed Compute and SimplePod
- Cloud installation tested and verified too
- Now the installer will download 30+ pre-trained demo voices
- Full preset save and load system implemented
- Loop all models feature implemented thus you can loop and generate with all models and see how they work
- Now all generations will be saved inside outputs folder with metadata
- Now batch folder processing works perfect just give input and output folder path, it will process every file in given folder and save with same file name in output folder
- Auto separate vocals + music and remix after conversion implemented thus with 1 click you can voice change full music audio files
- Save as MP3 feature added
- The parameters on the interface set to best as default
- Uses below 4 GB VRAM and ultra fast
- I have tested `Windows_Start_Realtime_Voice_Changing_Desktop_GUI.bat` and realtime voice change working too - taking speaking from your microphone and real time
- For offline inference and training use `Windows_Start_Voice_Change_And_Train_WebUI.bat`
- `Command_To_Test_RunPod_SimplePod_GPU.txt` added because out of 3 pods i rented, 2 had broken hardware so best to test
- Make sure to select CUDA 12.9 and 13 filter in RunPod selection due to outdated NVIDIA drivers
- Make a fresh installation with `Windows_Install_and_Update.bat`

<p align="center">
 <img height="600" alt="image" src="https://github.com/user-attachments/assets/92cd30bd-64eb-43c3-ad4c-fa0eee03e6de" />

</p>

<p align="center">
 <img  height="600" alt="screencapture-127-0-0-1-7865-2026-04-10-00_41_49" src="https://github.com/user-attachments/assets/9e1e67c5-885b-401c-b26a-becee5af9245" />
</p>

## **Windows Requirements**

- Python 3.10, FFmpeg, CUDA 13, C++, CUDNN 9.17.1, C++ tools, MSVC and Git
- If it doesn't work make sure to follow below tutorial step by step and install everything exactly as shown in this below tutorial
- [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)
- The requirements tutorial post fully updated : [https://www.patreon.com/posts/111553210](https://www.patreon.com/posts/111553210)

<p align="center">
 <img width="1313" height="634" alt="image" src="https://github.com/user-attachments/assets/2e99aca1-c53c-4606-867f-a083b9a5992a" />
</p>

## **Massed Compute (Recommend Cloud) :**

- Please register via this link : [https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute](https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute)
  - Use our coupon `SECourses`
  - Our coupon works on all GPUs now
    - H100 has amazing price and speed but you can use like RTX A6000 ADA as well
    - Full details here : [https://www.patreon.com/posts/26671823](https://www.patreon.com/posts/26671823)
  - Then select our image `SECourses` from Creator dropdown
  - Then follow `Massed_Compute_Instructions_READ.txt`
  - Same as my any other Massed Compute installer script
  - Example tutorial for learn how to install and use Massed Compute
    - (Starts at 12:58) : [https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778](https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778)

## **RunPod (Cloud):**

- Please register via this link : [https://get.runpod.io/955rkuppqv4h](https://get.runpod.io/955rkuppqv4h)
  - Then follow `RunPod_SimplePod_Trellis_Instructions_READ.txt`
  - Same as my any other RunPod installer script
  - Use the template written in `RunPod_SimplePod_RVC_Instructions_READ.txt` file
  - Example tutorial for learn how to install and use RunPod
    - (starts at 22:03) : [https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323](https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323)
  - Template : [https://get.runpod.io/SECourses_CU13](https://get.runpod.io/SECourses_CU13)

## App Screenshots

<p align="center">
 <img height="600" alt="image" src="https://github.com/user-attachments/assets/520914a6-aa28-4a9a-b46b-fe739c585b97" />
</p>

<p align="center">
 <img  height="600" alt="image" src="https://github.com/user-attachments/assets/7bbf40e2-f4de-49da-b391-1dfb4f252558" />
</p>

<p align="center">
  <img  height="600" alt="image" src="https://github.com/user-attachments/assets/b6985e30-1396-4a5f-9930-890d3325999c" />

</p>

<p align="center">
   <img  height="600" alt="image" src="https://github.com/user-attachments/assets/43a69d2f-8176-48a0-add9-a1b5b2ef4106" />

</p>

<p align="center">
 <img  height="600" alt="image" src="https://github.com/user-attachments/assets/9e936edf-0e97-4c39-8649-28dad01085ef" />

</p>

