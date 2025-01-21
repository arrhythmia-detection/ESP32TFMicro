<h1 align="center">ESP32TFLMWrapper</h1>

<h6 align="center">
    <div align="center">
        <a href="https://www.espressif.com">
            <img src="https://img.shields.io/badge/espressif32-E7352C.svg?style=for-the-badge&logo=espressif&logoColor=white"  alt="whatever"/>
        </a>
        <a href="https://github.com/espressif/arduino-esp32">
            <img src="https://img.shields.io/badge/ESP32 Arduino Framework-00979D?style=for-the-badge&logo=Arduino&logoColor=white"  alt="whatever"/>
        </a>
        <a href="https://www.tensorflow.org/lite">
            <img src="https://img.shields.io/badge/TFLITE-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"  alt="whatever">
        </a>
        <a href="https://github.com/tensorflow/tflite-micro">
            <img src="https://img.shields.io/badge/TFLITE micro-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"  alt="whatever">
        </a>
    </div>


<h6 align="center"> 
        <a href="https://github.com/arrhythmia-detection/ESP32TFMicro/actions/workflows/tflm_wrapper_integration_test.yml">
            <img src="https://github.com/arrhythmia-detection/ESP32TFMicro/actions/workflows/tflm_wrapper_integration_test.yml/badge.svg"  alt="whatever">
        </a>
        <a href="https://github.com/arrhythmia-detection/ESP32TFMicro/actions/workflows/pio_registry_publisher.yml">
            <img src="https://github.com/arrhythmia-detection/ESP32TFMicro/actions/workflows/pio_registry_publisher.yml/badge.svg"  alt="whatever">
        </a>
        <div align="center">
            <hr width="250px"/>
            <a href="https://registry.platformio.org/libraries/inmoresentum/ESP32TFLMWrapper">
                <img src="https://badges.registry.platformio.org/packages/inmoresentum/library/ESP32TFLMWrapper.svg" alt="PlatformIO Registry" />
            </a>
            <hr width="250px"/>
        </div>
</h6>

</h6>

A simple [PlatformIO](https://platformio.org/) Arduino library that
abstracts away all the low level complexity of TensorFlow
Lite Micro and makes it really easy to deploy
TFLM (*T*ensor*F*low *L*ite *M*icro) models on supported ESP32 boards.
On top of that it adds the ability to print
per [ops](https://www.tensorflow.org/api_docs/cc/namespace/tensorflow/ops) wise
execution time (*latency*).

### Ops Execution Time Logging

By default, the library will log per ops wise execution time.
To stop logging the execution time, please add/append the build flag shown below
in your [platformio.ini](platformio.ini) file and build the project.

```shell
    build_flags = -DEXCLUDE_OPS_EXECUTION_TIME_LOGGING
```

### Acknowledgement

This library internally utilizes
[EloquentTinyML](https://github.com/eloquentarduino/EloquentTinyML)
and [tflm_esp32](https://github.com/eloquentarduino/tflm_esp32) developed
and maintained by [eloquentarduino](https://github.com/eloquentarduino).


> [!NOTE]
> For examples on how to use this library, please check out [examples](examples) folder.


&#160;

<p align="center">Copyright &copy; 2024-present 
   <a href="https://github.com/Inmoresentum" target="_blank">Inmoresentum</a>
    and Contributors.
</p>

<h6 align="center">
   <a href="LICENSE">
      <img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=License&message=MIT&colorA=AFA5FA&colorB=FF60B4"
         alt="whatever" style="border-radius: 5px"/>
   </a>
</h6>
