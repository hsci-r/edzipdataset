from pathlib import Path
import sqlite3
from typing import Sequence, cast

import pytest
from torch import Tensor
from torch.utils.data import Dataset

from hscitorchutil.dataset import remove_nones_from_batch

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
    db: SQLiteDataset[int, str, tuple[int, str]] = SQLiteDataset(
        dbname, "test", "entry_number", "value, id", "id")
    assert len(db) == 3
    assert db[0] == (100, 'foo')
    assert db[1] == (200, 'bar')
    assert db[2] == (300, 'barfoo')
    assert db.__getitems__([0, 1]) == [(100, 'foo'), (200, 'bar')]
    assert db['foo'] == (100, 'foo')
    assert db['bar'] == (200, 'bar')
    assert db['barfoo'] == (300, 'barfoo')
    assert db.__getitems__(['foo', 'bar']) == [(100, 'foo'), (200, 'bar')]


def my_collate(batch) -> tuple[Tensor, tuple[str]]:
    return cast(tuple[Tensor, tuple[str]], remove_nones_from_batch(batch))


def test_sqlitedatamodule(dbname: str, tmp_path):
    from hscitorchutil.sqlite import SQLiteDataModule
    db = SQLiteDataModule(dbname, dbname, dbname, str(
        tmp_path / "cache"), "test", "entry_number", "value, id", "id", batch_size=2, collate_fn=my_collate)
    db.prepare_data()
    db.setup('test')
    assert len(db.test_dataloader()) == 2
    i = db.test_dataloader().__iter__()
    vals, ids = i.__next__()
    assert len(vals) == 2
    assert len(ids) == 2
    assert vals[0] == 100
    assert vals[1] == 200
    vals, ids = i.__next__()
    assert len(vals) == 1
    assert len(ids) == 1
    assert vals[0] == 300
    db.setup("validate")
    assert len(db.val_dataloader()) == 2  # type: ignore
    assert len(db.test_dataloader()) == 2  # type: ignore
