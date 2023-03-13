# Unsupervised Anomaly Detection

## 1. Setup environment
- To prevent causing conflicted between packages, we should create a new environment for each algorithm
- For example, to install packages to run RegAD, we do following steps:
    ```
    # move to RegAD folder
    $ cd RegAD

    # create new environment named "regad_env"
    $ python -m venv regad_env

    # activate new environment
    $ source regad_env/bin/activate

    # Install packages for RegAD
    $ pip install -r requirements.txt
    ```

## 2. Run algorithm to detect anomalies on webcam
To run RegAD using webcam:
```
$ python inference_cam.py --camidx 0 --config config.yaml --threshold 0.5
```

where:
- **camidx**: camera index (default: 0).

- **config**: path of config.yaml file, where model will load its configuration (if you use gpu, change "cpu" to "cuda" in DEVICE)

- **threshold**: threshold for model to decide whether it's defected or not 

