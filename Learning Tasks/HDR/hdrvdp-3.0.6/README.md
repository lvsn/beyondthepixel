HDR-VDP-3: A calibrated visual metric for visibility and quality
predictions in all luminance conditions

This repository contains Matlab code of the HDR-VDP-3 - a visual
difference predictor for high dynamic range images. This is the
successor of the original HDR-VDP and HDR-VDP-2.

The previous version of the metric (version 2.2.2) can be found at http://hdrvdp.sourceforge.net/

The current version number and the list of changes can be found in the ChangeLog.txt.

# Changes 

Compared to version 2.2.2 of the metric, the current version contains the following major changes:

* HDR-VDP-3 now requires an additional parameter `task` which controls the type of image comparison: 
  * `side-by-side` - side-by-side comparison of two images
  * `flicker` - the comparison of two images shown in the same place and swapped every 0.5 second
  * `detection` - detection of a single difference in a multiple-alternative-forced-choice task (the same task as in HDR-VDP-2)
  * `quality` - prediction of image quality
  * `civdm` - contrast invariant visual difference metric that can compare LDR and HDR images
* civdm is a new experimental metric based on the ideas from [7]. See examples/compare_hdr_vs_tonemapped.m
* CSF function has been refitted to a newer data (from 2014)
* MTF can be disabled or switched to the CIE99 Glare Spread Function
* The metric now accounts for the age-related effects, as described in [2]
* The metric includes a model of local adaptation from [3]
* The tasks `side-by-side` and `flicker` have been calibrated on large datasets from [4] and [5].
* The task `quality` has been recalibrated using a new UPIQ dataset with other 4000 SDR and HDR images, all scaled in JOD units. 
* Added multiple examples in the `examples` folder
* The code has been re-organized and tested to run on a recent version of Matlab (2019a)

# Installation

To install the metric just add the hdrvdp directory to the Matlab path. 

HDR-VDP requires matlabPyrTools (http://www.cns.nyu.edu/~lcv/software.html).
The first invocation of the hdrvdp3() function will add matlabPyrTools 
automatically to the Matlab path. If you already have matlabPyrTools in 
the path, the metric may fail, as HDR-VDP-3 requires a patched version of 
that toolbox. 

# Using the metric

To run the metric:

Please check the "examples" folder and consult also the documentation for hdrvdp3.m before running the metric. 

The metric accounts for many more factors than typical image quality metrics, such as PSNR or SSIM. Because of that, the metric requires fairly precise specification of viewing conditions: display resolution, viewing distance, display peak brightness. HDR images must be provided in the absolute photometric units (cd/m^2) to account for the screen brightness. If those paramaters are not set correctly, the predictions are likely to be inaccurate. 

# References

The paper describing version 3 of the metric is in preparation. In the meantime, please refer to the original HDR-VDP-2 paper: 

[1] Mantiuk, R., Kim, K. J., Rempel, A. G., & Heidrich, W. (2011). 
HDR-VDP-2: A calibrated visual metric for visibility and quality predictions in all luminance conditions. 
ACM Transactions on Graphics, 30(4), 40:1--40:14. 
https://doi.org/10.1145/2010324.1964935

If you use the metric in your work/paper, please cite the above paper 
AND the version of the metric you used, for example "HDR-VDP 3.0.6". Check
ChangeLog.txt or hdrvdp_version() for the current version of the metric.

The current version includes the age-adaptive components described in:

[2] Mantiuk, R. K., & Ramponi, G. (2018). 
Age-dependent predictor of visibility in complex scenes. 
Journal of the Society for Information Display, 1–21. 
https://doi.org/10.1002/jsid.623

and a model of local adaptation from:

[3] Vangorp, P., Myszkowski, K., Graf, E. W., & Mantiuk, R. K. (2015). 
A model of local adaptation. 
ACM Transactions on Graphics, 34(6), 1–13. 
https://doi.org/10.1145/2816795.2818086


The tasks `side-by-side` and `flicker` have been calibrated on a large datasets from:

[4] Wolski, K., Giunchi, D., Ye, N., Didyk, P., Myszkowski, K., Mantiuk, R., … Mantiuk, R. K. (2018). 
Dataset and Metrics for Predicting Local Visible Differences. 
ACM Transactions on Graphics, 37(5), 1–14. 
https://doi.org/10.1145/3196493

and

[5] Ye, N., Wolski, K., & Mantiuk, R. K. (2019). 
Predicting Visible Image Differences Under Varying Display Brightness and Viewing Distance. 
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5429–5437. 
https://doi.org/10.1109/CVPR.2019.00558

The task `quality` has been calibrated on a new UPIQ dataset (publication in review), which has been scaled using the methods from: 

[6] Perez-Ortiz, M., Mikhailiuk, A., Zerman, E., Hulusic, V., Valenzise, G., & Mantiuk, R. K. (2019). 
From pairwise comparisons and rating to a unified quality scale. 
IEEE Transactions on Image Processing, 29, 1139�1151. 
https://doi.org/10.1109/tip.2019.2936103

The 'civdm' metric is experimental (paper in preparation) and based on the ideas from:

[7] Aydin, T. O., Mantiuk, R., Myszkowski, K., & Seidel, H.-P. (2008). 
Dynamic range independent image quality assessment. 
ACM Transactions on Graphics (Proc. of SIGGRAPH), 27(3), 69. 
https://doi.org/10.1145/1360612.1360668


# Calibration reports

The key feature of HDR-VDP is that it is calibrated and tested agains a large number of datasets. The peformance of the metric for the "detection" task can be checked at: https://gfxdisp.github.io/hdrvdp-reports/visibility_report/

# Help and contact

If possible, please post your question to the google group:
http://groups.google.com/group/hdrvdp

If the communication needs to be confidential, contact me
directly. Please include "[n0t5pam]" in the subject line so that your
e-mail is not filtered out by the SPAM filter).

Rafal Mantiuk <mantiuk@gmail.com>

For more more information, refer to the project web-site:
http://hdrvdp.sourceforge.net/
