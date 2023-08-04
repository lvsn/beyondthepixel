Display emission spectra
------------------------

Files emission_spectra_*cvs contain the spectrum of emitted light for a few standard displays. 

The first column contains wavelength in nm. The corresponding three columns contain emitted radiance when the display is showing the maximum brightness for red, green and blue primaries.

emission_spectra_oled.csv - measured for an OLED display in Oppo 9R smartphone
  - this is a wide color gamut display. It could be used to reproduce images rec.2020 color space.

emission_spectra_led-lcd-srgb.csv - measured for Acer BM320 IPS LCD display set to the sRGB color gamut
  - the primaries are very close to the rec.709 primaries

emission_spectra_led-lcd-wcg.csv - measured for Acer BM320 IPS LCD display set to the native color gamut
  - this is a wide color gamut display. It could be used to reproduce images rec.2020 color space.

emission_spectra_ccfl-lcd.csv - measured for Hazro HZ27WB 27" LCD with CCFL backlight
  - this is a wide color gamut display. It could be used to reproduce images rec.2020 color space.

The other emission spectra are historical. Their gamuts do not cover rec.709 so please use them with care. 


Cone fundamentals and CMF
-------------------------

Cone fundamentals and color matching functions are downloaded from:

http://cvrl.ucl.ac.uk/

