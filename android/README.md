# Android gradle project for building katago

## Building
Open the project in android studio, it was tested with Android Studio `2024.1.1 Patch 1`

### Obtaining libOpenCL.so
The headers used in this project in `app/src/main/cpp/include` come from https://github.com/krrishnarraj/libopencl-stub

libOpenCL.so is not commonly freely available, so it's easiest to copy it from a phone.
This works on most devices, like Xiaomi Redmi or Lenovo Tab, but it seems not to work with Google pixel 7a.

Copying the libOpenCL.so from your android device:
- enable developer options on your phone and enable usb or wifi debugging
- connect the phone to your development computer, it should show up in the device manager of android studio
- make sure you have adb, it can be install on ubuntu using `sudo apt install google-android-platform-tools-installer`
- open a terminal and do the following:
  ```
  cd <path to KataGo repo>/android/app/src/main/jniLibs
  adb pull /vendor/lib64/libOpenCL.so arm64-v8a/
  adb pull /vendor/lib/libOpenCL.so armeabi-v7a/
  ```
  if adb pull says there are multiple devices you can obtain the device id and add -s to the command like so: 
  ```
  adb devices -l
  $ adb devices -l
  List of devices attached
  23011WERA0ADER         device 4-1 product:lynx model:Pixel_7a device:lynx transport_id:192
  adb-HA1KX6ED-J5mLEN._adb-tls-connect._tcp device product:TB-J616F_EEA model:Lenovo_TB_J616F device:TB-J616F transport_id:78
  adb -s adb-HA1KX6ED-J5mLEN._adb-tls-connect._tcp pull /vendor/lib64/libOpenCL.so arm64-v8a/ arm64-v8a/
  adb -s adb-HA1KX6ED-J5mLEN._adb-tls-connect._tcp pull /vendor/lib/libOpenCL.so armeabi-v7a/ armeabi-v7a/
  ```