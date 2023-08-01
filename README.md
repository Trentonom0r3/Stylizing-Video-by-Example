This is a [Partial] Python Implementation of the Paper "Stylizing Video by Example" by Jamriska et al.
Currently, stylization is only based on a single keyframe. I'll be updated to use multiple at a later time. 
Instead of SIFT flow like in the original paper, I'm using RAFT optical flow, and PhyCV edge detection to get the edges.

## Usage
- Clone my Ebsynth Fork and build the executable. 
- Clone this Repo.
- Move the Ebsynth.exe file into the same folder as ```main.py```
- Run!

```
python main.py <style_file> <input_dir>
```

OR

```
python main.py <style_file> <input_dir> [--flow_quality {low,high}] [--edge_method {page,classic,pst}]
```

- flow_quality low uses opencv farneback. Faster, but lower accuracy. Default Option.
- flow_quality high uses RAFT. Slower, but much better accuracy.

- edge_method options
  - PAGE : Directional Edge Detection. Runs on GPU, takes less than a minute to run. Most Accurate of the 3. Tends to have great fine details as well as overall structure. This is the Default Option.
  - Classic: Gaussian Edge Detection as Defined in the original paper. Runs on CPU, seconds to run. Accurate in some ways, inaccurate in others. This is a good middle-ground. 
  - PST: Physics based "canny" edge detection. Runs on GPU, takes less than a minute. This produces great structural results, but fine details tend to be lost.

```
pip install phycv 
```

## Demo



https://github.com/Trentonom0r3/Stylizing-Video-by-Example/assets/130304830/567c1317-d991-4284-9a6a-80518e784686

## To do:

- Modify Ebsynth to accept cv::mat instead of file paths, so we only have to save the output files.
- Add Multi-Keyframe support
- Add extra Temporal Smoothing/Denoising Step
- [This will take a bit of exploration] The ability to start with style frames, and raw flow data from 3d modeling software.
- Make things more modular, easier to change and use in other things. (I want to use this in my AE extension, and I'd like it to be easily accessed by anyone else, I think It could lead to some cool extensions for SD and/or touchdesigner)
- add SIFT flow option, command line args to make it easier to choose different methods for guides


# RAFT
This repository contains the source code for our paper:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

<img src="RAFT.png">

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```

## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
