"# exam-ai" 
build image 
docker build -t fastapi-cv-app .

run app:
docker run -d -p 8000:8000 fastapi-cv-app
