from collections.abc import Iterator
from pathlib import Path

from fastapi import Request
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def create_database_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    is_sqlite = url.get_backend_name() == "sqlite"

    if not is_sqlite:
        return create_engine(database_url)

    # Make sure the folder for the SQLite file exists.
    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)

    # FastAPI runs sync routes in a thread pool, so connections must be
    # usable from threads other than the one that opened them.
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """SQLite ignores foreign keys unless this is turned on for each connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_tables(engine: Engine) -> None:
    """Create any missing tables, then add any columns added since. Alembic replaces this later."""
    from app.db import models  # noqa: F401  (importing registers the tables on Base)

    Base.metadata.create_all(engine)
    add_columns_missing_from_existing_tables(engine)


# Columns added after the first release. create_all() only creates missing
# tables, so an existing database needs these added by hand.
COLUMNS_ADDED_LATER = [
    ("features", "archived_at", "DATETIME"),
    ("submissions", "archived_at", "DATETIME"),
]


def add_columns_missing_from_existing_tables(engine: Engine) -> None:
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, column_name, column_type in COLUMNS_ADDED_LATER:
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))


def get_request_db_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency: opens one database session for the request."""
    with request.app.state.session_factory() as session:
        yield session
