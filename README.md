# LaTTe: LAnguage Trajectory TransformEr

<video width="100%" controls>
  <source src="./docs/media/ICRA2023_LaTTe_low.mp4" type="video/mp4"/>
</video>
<!-- ![iterative NL interactions over a trajectory](./docs/media/interactions.gif)
-->


## setup
<sub>_tested on Ubuntu 18.04 and 20.04_</sup>

[install anaconda](https://docs.anaconda.com/anaconda/install/linux/)

Environment setup
```
conda create --name py38 --file spec-file.txt python=3.8
conda activate py38
```
Install CLIP + opencv
```
pip install ftfy regex tqdm dqrobotics rospkg similaritymeasures Cython
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
```

Download models

```
pip install gdown=4.6.1
gdown --folder https://drive.google.com/drive/folders/1r8BYvpu1AMj9tY0gt5YYigYgZqhzHQhf?usp=sharing -O models/.
```
Download synthetic dataset  
```
gdown --folder https://drive.google.com/drive/folders/11NAmB1Rma-gOsh-b-KZB90xjGplal2Z8?usp=sharing -O data/.
```

Download image dataset(optional)
```
gdown --id 1AyWxJ9SSjWML6rOgokFsxOjaU2BEUP9q -O image_data.zip
unzip image_data.zip -o image_data/.
```

Configure the paths at [src/config.py](src/config.py) 

## Animated demo

follow the notebook [interactive_user_study.ipynb](interactive_user_study.ipynb)

---
---

## Running the visual demo - Deprecate

```
cd src
python interactive.py
```

**How to use:**

1) press 'o' to load the original trajectory
2) press 'm' to modify the trajectory using our model for the given input on top.
3) press 't' to set a different interaction text.
4) press 'u' to update the trajectory setting the modified traj as the original one

instructions for additional keyboard commands are shown in the script output.

---
## ROS setup:

> **IMPORTANT:** Make sure that conda isn't initialized in your .bashrc file, otherwise, you might face conflicts between the Python versions 

[install ROS melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)

<!-- [manually install CVbridge](https://cyaninfinite.com/ros-cv-bridge-with-python-3/)
> **NOTE:** this is the catkin config that I used to install CVbridge with the Anaconda </br>
```catkin config -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python3.8 -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython3.8.so -DSETUPTOOLS_DEB_LAYOUT=OFF``` -->

For realtime object detection:
```
git clone https://github.com/arthurfenderbucker/realsense_3d_detector.git
```

## Running with ROS
terminal 1
```
roscore
```
terminal 2
```
roscd latte/src
python interactive.py --ros true
```

---

## coppelia_simulator + ROS + anaconda setup
install coppelia simulator
https://www.coppeliarobotics.com/helpFiles/en/ros1Tutorial.htm
add ```export COPPELIASIM_ROOT_DIR=~/path/to/coppeliaSim/folder``` to your ~/.bashrc

```
cd <ros_workspace>/src
git clone https://github.com/CoppeliaRobotics/ros_bubble_rob
git clone --recursive https://github.com/CoppeliaRobotics/simExtROS.git sim_ros_interface
cd <ros_workspace>
```

```
catkin config -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python3.8 -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython3.8.so -DSETUPTOOLS_DEB_LAYOUT=OFF

catkin config --install
catkin build
```

## Other relevant files
overview of the project
[model_overview.ipynb](model_overview.ipynb)

model variations and ablation studies
[Results.ipynb](Results.ipynb)

user study interface
[user_study.py](user_study.ipynb)

generate syntetic dataset
[src/data_generator_script.py](src/data_generator_script.py)


