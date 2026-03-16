# Abstract

Проект для обучения авторегрессионной модели для генерации анекдотов. Модель будет обернута в контенйер и размещена на плате raspberry 5.

# Host branch

First add file `tgtoken.py` with `TOKEN=<YOUR_TOKEN>`

Build image:
```bash
docker build -t tgjokes:host -f Dockerfile.dev .
```

Run container:
```bash
docker run -d -it --name tgjoke_host_container --restart unless-stopped --network host -v $(pwd):/workspace tgjokes:host
```

Here `--restart unless-stopped` automatically run container if it was stopped

Here `--network host` need for run container with proxy

For stop and remove container:
```bash
docker stop tgjoke_host_container 
docker remove tgjoke_host_container 
```

For check logs in container:
```bash
docker logs tgjoke_host_container
```