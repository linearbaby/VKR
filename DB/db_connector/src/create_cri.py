import os


def create_cri(
    dbapi="mysql+pymysql",
    user="root",
    password="password",
    host="127.0.0.1",
    port="3306",
    db_name="flask",
):
    dbapi = os.getenv("GRAD_DBAPI", dbapi)
    user = os.getenv("GRAD_USER", user)
    password = os.getenv("GRAD_PASSWORD", password)
    host = os.getenv("GRAD_HOST", host)
    port = os.getenv("GRAD_PORT", port)
    db_name = os.getenv("GRAD_DB_NAME", db_name)

    return f"{dbapi}://{user}:{password}@{host}:{port}/{db_name}"
    "mysql+pymysql://root:password@127.0.0.1:3306/flask"
