
# Study on our Results: Estimated FPS and Actual FPS  

Here, we will study our experiment results whether those are reasonable based on theory and calculation.  


## Environment  

Author's environment:
>ARM.
> A Qualcomm Snapdragon 810. We use a highly-optimized Neon-based
> implementation. A single thread is used for evaluation.
([ShuffleNet V2: Practical Guidelines for Efficient
CNN Architecture Design](https://arxiv.org/pdf/1807.11164.pdf) p.3)  

Author uses Qualcomm Snapdragon 810. Therefore, we estimated their GFLOPS as around 300 from below webpage.  
> 324~388.8GFLOPS  
(https://gpuflops.blogspot.com/2015/02/gpu-flops-list.html?m=1)  


My environment([source](https://www.raspberrypi.org/products/raspberry-pi-2-model-b/))　:
>  A 900MHz quad-core ARM Cortex-A7 CPU  
>  1GB RAM

We checked Pi FLOPs using ```lscpu```.  
> pi@raspberrypi:~ $ lscpu  
> Architecture:          armv7l  
> Byte Order:            Little Endian  
> CPU(s):                4  
> On-line CPU(s) list:   0-3  
> Thread(s) per core:    1  
> Core(s) per socket:    4  
> Socket(s):             1  
> Model:                 4  
> Model name:            ARMv7 Processor rev 4 (v7l)  
> CPU max MHz:           624.0000  
> CPU min MHz:           600.0000  
> BogoMIPS:              38.40  
> Flags:                 half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm crc32  

At Flags, you can see neon is activated.  
From above, we calculated our Pi GFLOPS as follows.  

> Raspberry Pi 2 	Cortex-A7 	4.8 GFLOPS 	0.6GHz 	4 	8 	NEON: 2(mad) x1(simd) x4(core) x0.6(clock) = 4.8 GFLOPS  
(I referred here https://dench.flatlib.jp/opengl/cpuflops to derive estimation.)  

## Results
*FPS is execution time of forward pass per image.  

1.0 ShufflenetV2: FPS is 1.09 and Model size is 5.3MB  
```python 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.000
```

1.0 MobilenetV2: FPS is 0.66 and Model size is 9.3MB
```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.045
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.267
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.052
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.115
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.500
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.115
 ```
 
 Resnet18: FPS is 0.39 and Model size is 44.9MB
 ```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.257
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.642
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.208
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.269
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.410
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.800
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.400
 ```

## Comparison  

1.0 Shufflenet V2:  
From [paper](https://arxiv.org/pdf/1807.11164.pdf), I will calculate estimated FPS.  
24.4x(4.8GFLOPS/324GFLOPS)=0.36  

The estimated FPS is 0.36.  
However, this is just a third of actual FPS(1.09).  


1.0 Mobilenet V2:  
8.9x(4.8GFLOPS/324GFLOPS)=0.13  

However, observed result is 0.66  
This is a fifth of it.  

We are trying to find this reason.  
