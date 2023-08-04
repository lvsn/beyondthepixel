# HDR Dataset calibration

The scripts here are made to calibrate an omnidirectional camera covering an hemisphere (180°) with the corresponding illuminance values.
The input image are assumed to be of HDR format (.exr) in linear color space


## Requirements

The contrib version of opencv is required for the omnidirection module used for the camera:

```sh
pip install opencv-contrib-python
```

## Geometric calibration

The first step to calibrating the dataset is the geometric calibration of the camera.

```sh
python CalibrateGeo.py --input_rep ./imgs --geometricCalib_file GeoCalib.pkl --checkboard_x 10 checkboard_y 7
```

## Photometric calibration

The camera can now be calibrated for its photometry

```sh
python PhotometricCalib.py --input_rep ./imgs --geometricCalib_file GeoCalib.pkl
```

The corresponding illuminance of an image is expected to have the same basename in the input_rep directory with the .csv extension storing (X,Y,Z) values


## Notes

The scripts may contain minor errors because they were cleaned up for easier understanding