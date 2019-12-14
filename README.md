Simple VAE experiment on a mixture-of-gaussians dataset.
Refer to VAE_Tutorial.pdf for a lot of details.

To run this code:
```
$ python3 generateData.py <fileName> <numPoints>
$ python3 plotData.py <fileName>
$ python3 gaussianModel.py <fileName> <epochs>
$ python3 sampleFromModel.py <pickleFile> <numPoints>
```
