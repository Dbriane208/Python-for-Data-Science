FROM python:3.10
EXPOSE 5000
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install flask pandas scikit-learn
COPY . .
CMD [ "flask","run","--host","0.0.0.0" ]