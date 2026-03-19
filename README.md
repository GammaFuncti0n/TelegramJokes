# Abstract

Telegram bot for joking in chat. It contain /generate function for make joke from scratch or from prompt. Also during chating, bot able with some probability joke on some message.

# Quick start

First add file `configs/token.yaml` with `TOKEN: <YOUR_TOKEN>`

Build image:
```bash
docker build -t tgjokes:dev -f Dockerfile.dev .
```

Run container:
```bash
docker run -d -it --name tgjoke_host_container --restart unless-stopped --network host -v $(pwd):/workspace tgjokes:dev
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