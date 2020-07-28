import pytest
from pathlib import Path

from rydanalysis import OldStructure


def test_is_dir(tmpdir):
    no_dir_path = Path(tmpdir) / 'something_not_meaningful'
    with pytest.raises(AttributeError):
        OldStructure(no_dir_path)


def test_empty_directory(empty_structure):
    assert empty_structure.tmstps == []
    assert empty_structure.images is None
    assert empty_structure.traces is None


def test_update_few_tmstps(update_few):
    old_structure = update_few
    for i in range(5):
        assert len(old_structure.tmstps) == 10 * (i + 1)
        new_tmstps = old_structure.update_tmstps()
        assert len(new_tmstps) == 10


def test_only_images(only_images):
    data = only_images.data
    assert data.shot.name == 'shot'
