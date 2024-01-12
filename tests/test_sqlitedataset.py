from pathlib import Path
import sqlite3

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="session")
def dbname(tmp_path_factory) -> str:
    db_path = str(tmp_path_factory.mktemp("data") / "test.db")
    con = sqlite3.connect(db_path)
    with con:
        con.execute(
            "CREATE TABLE test (entry_number INTEGER PRIMARY KEY, id VARCHAR(255), value INTEGER)")
        con.execute("INSERT INTO test VALUES (0, 'foo', 100)")
        con.execute("INSERT INTO test VALUES (1, 'bar', 200)")
        con.execute("INSERT INTO test VALUES (2, 'barfoo', 300)")
    return db_path


def test_sqlitedataset(dbname: str):
    from hscitorchutil.sqlite import SQLiteDataset
    db = SQLiteDataset(dbname, "test", "entry_number", "value, id", "id")
    assert len(db) == 3
    assert db[0] == (100, 'foo')
    assert db[1] == (200, 'bar')
    assert db[2] == (300, 'barfoo')
    assert db.__getitems__([0, 1]) == [(100, 'foo'), (200, 'bar')]
    assert db['foo'] == (100, 'foo')
    assert db['bar'] == (200, 'bar')
    assert db['barfoo'] == (300, 'barfoo')
    assert db.__getitems__(['foo', 'bar']) == [(100, 'foo'), (200, 'bar')]


def test_sqlitedatamodule(dbname: str, tmp_path):
    from hscitorchutil.sqlite import SQLiteDataModule
    db = SQLiteDataModule(dbname, dbname, dbname, str(
        tmp_path / "cache"), "test", "entry_number", "value, id", "id", batch_size=2)
    db.prepare_data()
    db.setup()
    assert len(db.train_dataloader()) == 2
    vals, ids = db.train_dataloader().__iter__().__next__()
    assert len(vals) == 2
    assert len(ids) == 2
    assert len(db.val_dataloader()) == 2  # type: ignore
    assert len(db.test_dataloader()) == 2  # type: ignore
    assert len(db.predict_dataloader()) == 5  # type: ignore
