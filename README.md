# Abstract

Проект для обучения авторегрессионной модели для генерации анекдотов. Модель будет обернута в контенйер и размещена на плате raspberry 5.

Датасет для обучения модели расположен в https://www.dropbox.com/scl/fi/z1ihoqwdj3zxil28ha86i/jokes.txt?rlkey=66y4gpclxmzimc86aa6agnjn8&st=mff21u6h&dl=0

# Environment

For build image:
```bash
docker build -t tgjoke:dev1 .
```

For run container:
```bash
docker run -it -d --name tgjoke_container -v $(pwd):/workspace tgjoke:dev1 
```

# Заметки
В чат надо будет написать начало анекдот, а модель его продолжит, можно будет попробовать добавить рекомендательную систему, когда модель предагает анекдот, а пользователь говорит нравится ему или нет.