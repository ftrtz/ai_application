# ai_application

RAYLYTIC CodingChallenge - see [documentation.ipynb](./documentation.ipynb)

## Run main.py with docker to generate and store results

Build docker image

```shell
$ docker build --tag ai_application .
```

Start docker container

```shell
$ docker-compose up
```

Results are stored in [/results](./results)