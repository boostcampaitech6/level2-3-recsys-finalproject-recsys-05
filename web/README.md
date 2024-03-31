# Duofinder Full-Stack Web Application


![python](https://img.shields.io/badge/3.12-gray?style=flat&logo=python&logoColor=white&label=Python&labelColor=3776AB)
![django](https://img.shields.io/badge/5.0.2-gray?style=flat&logo=django&logoColor=white&label=Django&labelColor=092E20)

## Requirements

- `python3.12`
- `poetry`


## Development Guide

```bash
git clone https://github.com/boostcampaitech6/level2-3-recsys-finalproject-recsys-05.git duofinder
cd duofinder/web
```

## 패키지 설치

```bash
poetry install
```

## 데이터베이스 초기화

sqlite로 기본 세팅됩니다.

```bash
poetry run python project/manage.py migrate
```

## 서버 실행

```bash
# 8000 포트로 개발 서버 실행
poetry run python project/manage.py runserver 8000
```
