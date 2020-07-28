import pytest
from pathlib import Path

from rydanalysis import OldStructure


class OldStructureOnly10(OldStructure):
    def update_tmstps(self):
        new_tmstps = super().update_tmstps()
        n_new = len(new_tmstps)

        if n_new > 10:
            self.tmstps = self.tmstps[:-(n_new-10)]
            return new_tmstps[:-(n_new-10)]
        else:
            return new_tmstps
        # variables_path = self.path / 'Variables'
        # new_tmstps = [tmstp for tmstp in self.iter_tmstps(variables_path, self.filename_pattern + '.txt')
        #              if tmstp not in self.tmstps][:2]
        # self.tmstps.extend(new_tmstps)
        # return new_tmstps


@pytest.fixture(scope="session")
def empty_structure(tmpdir_factory):
    path = tmpdir_factory.mktemp("empty_structure")
    return OldStructure(path)


@pytest.fixture(scope="session")
def only_images():
    path = Path.cwd()
    only_images_path = path.parent / "samples" / "old_structure" / "2020_07_06" / "05_SNR_EIT_AT_pABSx0-8_pBlue3-6"
    return OldStructure(only_images_path)


@pytest.fixture(scope="session")
def update_few():
    path = Path.cwd()
    only_images_path = path.parent / "samples" / "old_structure" / "2020_07_06" / "05_SNR_EIT_AT_pABSx0-8_pBlue3-6"
    return OldStructureOnly10(only_images_path)



if __name__ == '__main__':
    empty_structure()