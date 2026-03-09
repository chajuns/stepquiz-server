# 파이썬 3.9 환경 사용
FROM python:3.9-slim

# 작업 폴더 지정
WORKDIR /app

# 라이브러리 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 내 파이썬 코드 전체 복사
COPY . .

# 포트 8080 개방 및 서버 실행
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
