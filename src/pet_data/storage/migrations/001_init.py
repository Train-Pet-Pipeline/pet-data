"""Initial migration: create frames table and indexes."""
from pathlib import Path


def upgrade(conn):
    """Apply the initial schema: frames table and all indexes.

    Reads schema.sql from the parent storage package directory and executes it.
    Safe to call on an already-initialised database because the SQL uses
    ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS``.

    Args:
        conn: An open :class:`sqlite3.Connection`.
    """
    schema_path = Path(__file__).parent.parent / "schema.sql"
    conn.executescript(schema_path.read_text())


def downgrade(conn):
    """Tear down the initial schema by dropping the frames table.

    Args:
        conn: An open :class:`sqlite3.Connection`.
    """
    conn.execute("DROP TABLE IF EXISTS frames")
