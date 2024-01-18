
# Person Follower
This repository focuses on building a ROS2 based Person Follower.


## About the Project
This project aims to develop a comprehensive ROS2 simulation package tailored for a differential drive robot. The robot will be equipped with person-following capability, leveraging Kinect camera technology for person detection and tracking. The system's primary objective is to enable the robot to autonomously follow individuals, showcasing its potential for practical applications in various domains, including payload transportation and assistance tasks.

### Built With

* [![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://ubuntu.com/)
* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
* [![ROS 2](https://img.shields.io/badge/ros-%230A0FF9.svg?style=for-the-badge&logo=ros&logoColor=white)](https://www.sphinx-docs.org)
* [![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
* [![MediaPipe](https://img.shields.io/badge/mediapipe-%4285F4.svg?style=for-the-badge&logo=mediapipe&logoColor=white)](https://developers.google.com/mediapipe)
  

## Getting started with the project
Follow these instructions to set up this project on your system.

### Prerequisites

Follow these steps to install ROS Humble and OpenCV
* ROS Humble
Refer to the official [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html)

* OpenCV
  ```bash
  pip install opencv-contrib-python
  ```

### Installation

1. Make a new workspace
    ```bash
    mkdir -p personfollower_ws/src
    ```

2. Clone the ROS-Perception-Pipeline repository

    Now go ahead and clone this repository inside the "src" folder of the workspace you just created.

      ```bash
      cd personfollower_ws/src
      git clone git@github.com:atom-robotics-lab/Person-Follower.git
      ```

3. Compile the package

    Follow this execution to compile your ROS 2 package
  
      ```bash
      colcon build --symlink-install
      ```

4. Source your workspace
      ```bash
      source install/local_setup.bash
      ```

## Usage
### 1. Launch the simulation
We have made a demo world to test our person follower. To launch this world, follow the steps given below

```bash
ros2 launch person_follower_sim gazebo.launch.py
```
The above command will launch the world as shown below :

insert image of gazebo world

Don't forget to click on the **play** button on the bottom left corner of the Gazebo window.

### 2. Launch the Person follower node

## Testing

Now to see the inference results, open a new terminal and enter the given command

```bash
ros2 run rqt_image_view rqt_image_view
```
## Contributing

We wholeheartedly welcome contributions!  
They are the driving force that makes the open-source community an extraordinary space for learning, inspiration, and creativity. Your contributions, no matter how big or small, are **genuinely valued** and **highly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please adhere to this project's `code of conduct`.


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Contact Us

If you have any feedback, please reach out to us at:  
Our Socials - [Linktree](https://linktr.ee/atomlabs)
## Acknowledgments

* [Our wiki](https://atom-robotics-lab.github.io/wiki)
* [ROS Official Documentation](http://wiki.ros.org/Documentation)
* [Opencv Official Documentation](https://docs.opencv.org/4.x/)
* [Mediapipe Documentation](https://mediapipe.readthedocs.io/en/latest/)
* [Gazebo Tutorials](https://classic.gazebosim.org/tutorials)
* [Ubuntu Installation guide](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)
