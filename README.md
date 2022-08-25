# Object-aware Monocular Depth Prediction with Instance Convolutions

> https://arxiv.org/abs/2112.01521  
> https://ieeexplore.ieee.org/document/9726910  

> With the advent of deep learning, estimating depth from a single RGB image has recently received a lot of attention, being capable of empowering many different applications ranging from path planning for robotics to computational cinematography. Nevertheless, while the depth maps are in their entirety fairly reliable, the estimates around object discontinuities are still far from satisfactory. This can be contributed to the fact that the convolutional operator naturally aggregates features across object discontinuities, resulting in smooth transitions rather than clear boundaries. Therefore, in order to circumvent this issue, we propose a novel convolutional operator which is explicitly tailored to avoid feature aggregation of different object parts. In particular, our method is based on estimating per-part depth values by means of superpixels. The proposed convolutional operator, which we dub "Instance Convolution", then only considers each object part individually on the basis of the estimated superpixels. Our evaluation with respect to the NYUv2 as well as the iBims dataset clearly demonstrates the superiority of Instance Convolutions over the classical convolution at estimating depth around occlusion boundaries, while producing comparable results elsewhere.


## Installation
PyTorch 1.5, torchvision and scikit-image.

## Run the example code 
``` bash
$ python vanilla_net.py
```
## Citation

If you use this code for your research, please cite our paper:
```
@article{simsar2021object,
  author={Simsar, Enis and {\"O}rnek, Evin P{\i}nar and Manhardt, Fabian and Dhamo, Helisa and Navab, Nassir and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters}, 
  title={Object-Aware Monocular Depth Prediction With Instance Convolutions}, 
  year={2022},
  volume={7},
  number={2},
  pages={5389-5396},
  doi={10.1109/LRA.2022.3155823}
}

```
