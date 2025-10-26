# Proptimize



## Docker commands:


Build:
```sh 
docker build -t proptimize:1.0 .
```

Run:
```sh
docker run \ 
    -it \ 
    --gpus all \ 
    --shm-size 32GB \ 
    -e "WANDB_ENTITY=<WANDB_USER_NAME>" \ 
    -e "WANDB_API_KEY=<WANDB_API_KEY>" \ 
    -d \ 
    proptimize:1.0
```

View logs:
```sh
docker logs -f <CONTAINER ID>
```

List containers:
```sh
docker ps -a 
```