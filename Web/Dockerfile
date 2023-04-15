FROM base

COPY Web /app/Web
WORKDIR /app

ENTRYPOINT gunicorn -b 0.0.0.0:5000 "Web:create_app()"